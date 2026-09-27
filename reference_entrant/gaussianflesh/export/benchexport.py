"""GaussianBench submission writer for GaussianFlesh.

INTEGRATION PLAN (from inspecting the codebase, 2026-07):

  Simulation loop   gaussian_mesh_poc.GaussianBallSim.step() advances one
                    display frame = SUBSTEPS calls to solver_ul.substep()
                    followed by _finish_frame(), which runs plasticity,
                    updates _ti_covs = F @ _ti_rcov @ F^T, and syncs the
                    entity slice to numpy (sim.positions, sim.velocities,
                    sim.covs). DT and SUBSTEPS come from the --frame-dt and
                    --substep-dt CLI flags, parsed at module import.
  Particle state    module-level Taichi fields sliced per entity:
                    _ti_pos, _ti_vel, _ti_rest, _ti_F, _ti_C, _ti_m,
                    _ti_mu, _ti_lam, _ti_rcov (rest cov), _ti_covs
                    (deformed cov, what the renderer consumes), _ti_phase.
                    All f32 on CUDA (ti.init(arch=ti.cuda, default_fp=f32)).
  Display frame     the run harness (run_bench_scene.py) drives
                    solver_ul.substep(..., use_floor=False) directly plus
                    sim._finish_frame() per display frame, so NO simulator
                    source file needs modification, not even a hook call:
                    the recorder is fed from sim.positions / sim.velocities
                    / sim.covs after each frame, one device-to-host copy
                    per array per frame (done by _finish_frame itself).
  Renderer          gs_render.GSRenderer consumes positions + covariances;
                    used only when render recording is enabled.

This module SERIALIZES ONLY. It never imports the simulator or the
benchmark package; the run harness passes plain numpy arrays in. If a value
is wrong, it is exported wrong. The one geometric bookkeeping step is the
inverse of the harness's documented rigid placement transform (a constant
offset, optionally a fixed axis permutation for z-up scenes), so that the
submission is in scene coordinates exactly as SPEC.md requires; the
transform is recorded in meta.json notes.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SPEC_VERSION = "1.0.0"
SYSTEM_NAME = "GaussianFlesh"


def git_commit_hash(repo_dir: Path) -> str:
    """Current commit of the GaussianFlesh repo, or 'unknown'."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_dir),
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def hardware_string() -> str:
    """GPU name via torch if available (already a GaussianFlesh dep for
    gs_render), else CPU/platform info."""
    try:
        import torch
        if torch.cuda.is_available():
            return f"{torch.cuda.get_device_name(0)}, {platform.processor()}"
    except Exception:
        pass
    return f"{platform.processor()} ({platform.platform()})"


class BenchRecorder:
    """Buffers per-frame particle state and writes a GaussianBench
    submission directory on finalize().

    Frames are buffered in host memory as float32 (the simulator's native
    precision, so no information is lost). The largest benchmark scene is
    about 8000 particles x 961 frames x 6 floats = 185 MB, well within RAM,
    so no temp-file spill path is needed; the buffer is a list of per-frame
    arrays copied once from the device.

    scene_to_sim: dict describing the harness's rigid placement transform,
        {"offset": [ox, oy, oz], "permutation": None or 3x3 list,
         "frame_velocity": None or [vx, vy, vz] (scene coords)}.
        sim(t) = P @ (scene(t) - v_frame * t) + offset: a constant
        placement offset, an optional fixed axis permutation, and an
        optional Galilean boost (simulating in the frame moving at
        v_frame keeps positions small when a scene drifts far, which
        matters at f32). All three are exact, invertible, and
        physics-preserving; the recorder applies the exact inverse on
        export so the submission is in scene coordinates, and the
        transform is recorded in meta.json notes.
    """

    def __init__(self, out_dir: str | Path, scene_json_path: str | Path,
                 record_velocities: bool = True, record_mass: bool = True,
                 record_phase: bool = False, record_covariances: bool = False,
                 record_renders: bool = False,
                 scene_to_sim: dict | None = None,
                 notes: list[str] | None = None):
        self.out_dir = Path(out_dir)
        with open(scene_json_path, "r", encoding="utf-8") as f:
            self.scene = json.load(f)
        self.record_velocities = record_velocities
        self.record_mass = record_mass
        self.record_phase = record_phase
        self.record_covariances = record_covariances
        self.record_renders = record_renders
        self.notes: list[str] = list(notes) if notes else []

        st = scene_to_sim or {"offset": [0.0, 0.0, 0.0], "permutation": None}
        self._offset = np.asarray(st["offset"], dtype=np.float64)
        perm = st.get("permutation")
        self._perm_inv = (np.linalg.inv(np.asarray(perm, dtype=np.float64))
                          if perm is not None else None)
        fv = st.get("frame_velocity")
        self._frame_vel = (np.asarray(fv, dtype=np.float64)
                           if fv is not None else None)
        if (np.any(self._offset != 0.0) or self._perm_inv is not None
                or self._frame_vel is not None):
            self.notes.append(
                "Simulation ran in a transformed frame: "
                "sim(t) = P @ (scene(t) - v_frame * t) + offset, "
                f"offset={self._offset.tolist()}, "
                f"permutation={'identity' if perm is None else perm}, "
                f"v_frame={'zero' if fv is None else list(fv)} (Galilean "
                "boost; physics is Galilean-invariant and the boost keeps "
                "f32 positions small over a long drift). The exact inverse "
                "was applied on export; the submission is in scene "
                "coordinates.")

        self._frames: list[int] = []
        self._times: list[float] = []
        self._pos: list[np.ndarray] = []
        self._vel: list[np.ndarray] = []
        self._phase: list[np.ndarray] = []
        self._covs: dict[int, np.ndarray] = {}
        self._mass: np.ndarray | None = None
        self._n_particles: int | None = None
        self._t_start = time.perf_counter()
        self._render_dir_made = False

    # ------------------------------------------------------------------
    def _to_scene_frame(self, sim_pos: np.ndarray,
                        sim_time: float) -> np.ndarray:
        out = sim_pos.astype(np.float64) - self._offset
        if self._perm_inv is not None:
            out = out @ self._perm_inv.T
        if self._frame_vel is not None:
            out = out + self._frame_vel * sim_time
        return out

    def _vec_to_scene_frame(self, sim_vec: np.ndarray) -> np.ndarray:
        out = sim_vec.astype(np.float64)
        if self._perm_inv is not None:
            out = out @ self._perm_inv.T
        if self._frame_vel is not None:
            out = out + self._frame_vel
        return out

    def on_frame(self, sim_state: dict, frame_index: int,
                 sim_time: float) -> None:
        """Append one display frame. sim_state keys (numpy, host-side):
        positions (N, 3) required; velocities (N, 3); mass (N,);
        phase (N,); covariances (N, 3, 3). Arrays are copied, so callers
        may reuse buffers.
        """
        pos = np.asarray(sim_state["positions"])
        if self._n_particles is None:
            self._n_particles = len(pos)
        elif len(pos) != self._n_particles:
            raise ValueError(
                f"Frame {frame_index} has {len(pos)} particles, previous "
                f"frames had {self._n_particles}. Particle count must be "
                "constant.")
        self._frames.append(int(frame_index))
        self._times.append(float(sim_time))
        self._pos.append(
            self._to_scene_frame(pos, sim_time).astype(np.float32))

        if self.record_velocities:
            if "velocities" not in sim_state:
                raise ValueError("record_velocities is on but sim_state has "
                                 "no 'velocities'.")
            self._vel.append(self._vec_to_scene_frame(
                np.asarray(sim_state["velocities"])).astype(np.float32))
        if self.record_mass and self._mass is None:
            self._mass = np.asarray(sim_state["mass"],
                                    dtype=np.float32).copy()
        if self.record_phase:
            self._phase.append(np.asarray(sim_state["phase"],
                                          dtype=np.int8).copy())
        if self.record_covariances:
            cov = np.asarray(sim_state["covariances"], dtype=np.float64)
            if self._perm_inv is not None:
                # sim = P @ scene, so Sigma_scene = P^-1 Sigma_sim P^-T
                Pi = self._perm_inv
                cov = np.einsum("ij,njk,lk->nil", Pi, cov, Pi)
            self._covs[int(frame_index)] = cov.astype(np.float32)

    def add_render(self, frame_index: int, image_rgb: np.ndarray,
                   name: str | None = None) -> None:
        """Save one rendered frame as PNG under renders/."""
        from PIL import Image
        renders = self.out_dir / "renders"
        renders.mkdir(parents=True, exist_ok=True)
        self._render_dir_made = True
        fname = name if name else f"frame_{frame_index:05d}.png"
        Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).save(
            renders / fname)

    def write_camera(self, camera: dict) -> None:
        renders = self.out_dir / "renders"
        renders.mkdir(parents=True, exist_ok=True)
        with open(renders / "camera.json", "w", encoding="utf-8") as f:
            json.dump(camera, f, indent=2)

    # ------------------------------------------------------------------
    def finalize(self, extra_meta: dict | None = None) -> Path:
        """Write trajectory.parquet, optional covariances.parquet, and
        meta.json. Returns the submission directory path."""
        if not self._frames:
            raise ValueError("finalize() called with zero recorded frames.")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        n = self._n_particles
        F = len(self._frames)
        pid = np.tile(np.arange(n, dtype=np.int64), F)
        cols = {
            "frame": np.repeat(np.asarray(self._frames, dtype=np.int64), n),
            "time": np.repeat(np.asarray(self._times, dtype=np.float64), n),
            "particle_id": pid,
        }
        pos = np.concatenate(self._pos, axis=0)
        cols["x"], cols["y"], cols["z"] = pos[:, 0], pos[:, 1], pos[:, 2]
        present = []
        if self.record_velocities:
            vel = np.concatenate(self._vel, axis=0)
            cols["vx"], cols["vy"], cols["vz"] = vel[:, 0], vel[:, 1], vel[:, 2]
            present += ["vx", "vy", "vz"]
        if self.record_mass and self._mass is not None:
            cols["mass"] = np.tile(self._mass, F)
            present += ["mass"]
        if self.record_phase and self._phase:
            cols["phase"] = np.concatenate(
                [p.astype(np.int8) for p in self._phase])
            present += ["phase"]
        pq.write_table(pa.table(cols), self.out_dir / "trajectory.parquet")

        if self.record_covariances and self._covs:
            cframes = sorted(self._covs)
            rows = {
                "frame": np.repeat(np.asarray(cframes, dtype=np.int64), n),
                "particle_id": np.tile(np.arange(n, dtype=np.int64),
                                       len(cframes)),
            }
            stacked = np.stack([self._covs[f] for f in cframes])
            for key, (i, j) in (("sxx", (0, 0)), ("sxy", (0, 1)),
                                ("sxz", (0, 2)), ("syy", (1, 1)),
                                ("syz", (1, 2)), ("szz", (2, 2))):
                rows[key] = stacked[:, :, i, j].ravel()
            pq.write_table(pa.table(rows),
                           self.out_dir / "covariances.parquet")

        meta = {
            "system_name": SYSTEM_NAME,
            "system_version": git_commit_hash(Path(__file__).parent.parent),
            "spec_version": SPEC_VERSION,
            "columns_present": present,
            "wall_clock_seconds": round(time.perf_counter() - self._t_start, 3),
            "hardware": hardware_string(),
            "notes": " | ".join(self.notes) if self.notes else "",
        }
        if extra_meta:
            meta.update(extra_meta)
        with open(self.out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return self.out_dir
