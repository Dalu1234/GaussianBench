"""Run representation-native GaussianBench scenes through GASP's released
Taichi pipeline.

The driver mirrors the released ``taichi_examples/demo/3d.py`` exactly: the
native teddy pseudomesh vertices (stable triangle soup, three vertices per
triangle) are simulated as particles by the released taichi_elements
``MPMSolver``, GASP's released per-frame triangle size-control rule
(``modify_positions`` with its released default threshold 1.0) runs after
every step, and triangle checkpoints are exported with the released
``Rescale.inverse`` into the same frame as the frozen C5 submission. Scene
metrics are then produced by the benchmark's released converters
(``adapters/gasp_triangle_export.py`` for triangle_metrics.csv); A6
body_metrics are computed from the native solver state directly (positions,
velocities, per-particle deformation gradients read from the released
solver's own fields).

Scenes:
  a6_freefall_v1        native gravity rotated to the scene's -z axis; body
                        metrics in native solver units (meters)
  c6_rigid_transport_v1 zero gravity, uniform native initial velocity chosen
                        so the exported COM path follows the scene's declared
                        direction
  c7_supported_settle_v1 body resting on the released domain floor under the
                        released default gravity, long rollout
  c8_impact_recovery_v1 soft-elastic drop (released E_scale parameter) onto
                        the released domain floor

Controlled faults (--fault): half_gravity (a6), gravity_left_on (c6, the
released default gravity is simply not zeroed), sand_material (c8, released
inelastic material).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import taichi as ti
import torch

REPO = Path(__file__).parents[1]
MODEL = REPO / "models" / "teddy" / "teddybear" / "gs_flat" / "pseudomesh_info" / "ours_30000"
GAUSSIANBENCH = Path(
    os.environ.get("GAUSSIANBENCH_ROOT", REPO.parents[2] / "gaussianbench")
).resolve()

from engine.mpm_solver import MPMSolver  # noqa: E402  (released taichi_elements)

FRAME_DT = 1e-2  # released demo step size


class Rescale:
    """Released Rescale from taichi_examples/demo/3d.py, verbatim semantics."""

    def __init__(self, scale, offset):
        self.min = None
        self.max = None
        self.scale = scale
        self.offset = np.array(offset)

    def fit(self, x):
        self.min = x.min(axis=0)
        self.max = x.max(axis=0)

    def transform(self, x):
        return self.scale * (x - self.min) / (self.max - self.min) + self.offset

    def inverse(self, x):
        return (x - self.offset) / self.scale * (self.max - self.min) + self.min


def load_native_vertices() -> np.ndarray:
    pts = torch.load(MODEL / "vertices.pt").cpu().numpy()
    pts[:, 1] = -pts[:, 1]
    pts = pts[:, [0, 2, 1]]
    return pts.astype(np.float64)


def run(scene_id: str, out_root: Path, fault: str) -> None:
    started = time.perf_counter()
    pts_model = load_native_vertices()
    n = len(pts_model)
    assert n % 3 == 0

    configs = {
        "a6_freefall_v1": dict(scale=0.3, offset=[0.35, 0.35, 0.55], frames=4,
                               frame_dt=0.04, gravity=(0.0, 0.0, -9.8),
                               velocity=None, E=1.0, material="elastic"),
        "c6_rigid_transport_v1": dict(scale=0.4, offset=[0.2, 0.25, 0.2],
                                      frames=40, frame_dt=FRAME_DT,
                                      gravity=(0.0, 0.0, 0.0), E=1.0,
                                      material="elastic", velocity="direction"),
        "c7_supported_settle_v1": dict(scale=0.4, offset=[0.3, 0.07, 0.3],
                                       frames=150, frame_dt=FRAME_DT,
                                       gravity=None, velocity=None, E=1.0,
                                       material="elastic"),
        "c8_impact_recovery_v1": dict(scale=0.4, offset=[0.3, 0.5, 0.3],
                                      frames=60, frame_dt=FRAME_DT,
                                      gravity=None, velocity=None, E=0.05,
                                      material="elastic"),
    }
    if scene_id not in configs:
        raise SystemExit(f"NOT SUBMITTED: {scene_id} is outside GASP's "
                         "representation-native contract")
    cfg = configs[scene_id]

    if fault == "half_gravity":
        assert scene_id == "a6_freefall_v1"
        cfg["gravity"] = (0.0, 0.0, -4.9)
    elif fault == "gravity_left_on":
        assert scene_id == "c6_rigid_transport_v1"
        cfg["gravity"] = None  # released default (0, -9.8, 0) stays enabled
    elif fault == "sand_material":
        assert scene_id == "c8_impact_recovery_v1"
        cfg["material"] = "sand"
    elif fault != "none":
        raise SystemExit(f"Unknown fault {fault}")

    scaler = Rescale(cfg["scale"], cfg["offset"])
    scaler.fit(pts_model)
    pts_sim = scaler.transform(pts_model)

    velocity = None
    scene_json = json.loads(
        (GAUSSIANBENCH / "scenes" / f"{scene_id}.json").read_text())
    if cfg["velocity"] == "direction":
        d = np.asarray(
            scene_json["initial_conditions"]["uniform_velocity_direction"],
            float)
        # Exported velocity is D @ v_solver with D the released inverse-map
        # Jacobian, so pick v_solver = D^-1 d to realize the declared
        # direction exactly in the exported frame.
        D = (scaler.max - scaler.min) / cfg["scale"]
        v = d / D
        v = 0.25 * v / np.linalg.norm(v)
        velocity = [float(x) for x in v]

    ti.init(arch=ti.gpu, device_memory_fraction=0.6)
    material = {"elastic": MPMSolver.material_elastic,
                "sand": MPMSolver.material_sand}[cfg["material"]]
    mpm = MPMSolver(res=(64, 64, 64), E_scale=cfg["E"])
    if velocity is not None:
        mpm.add_particles(particles=pts_sim, material=material,
                          velocity=velocity)
    else:
        mpm.add_particles(particles=pts_sim, material=material)
    if cfg["gravity"] is not None:
        mpm.set_gravity(cfg["gravity"])

    threshold = 1.0  # released default
    init_scales = ti.field(dtype=ti.f32, shape=n)
    new_scales = ti.field(dtype=ti.f32, shape=n)

    def calc_scales(x):
        s = x.reshape((-1, 3, 3))
        s = s - np.expand_dims(s[:, 0, :], -2)
        return np.linalg.norm(s, axis=-1).flatten()

    @ti.kernel
    def modify_positions():
        # Released GASP triangle size-control rule (3d.py), verbatim.
        for i in range(mpm.n_particles[None] // 3):
            m = mpm.x[3 * i]
            for j in ti.static(range(1, 3)):
                idx = 3 * i + j
                diff = mpm.x[idx] - m
                norm = diff.norm()
                v = diff / norm
                if new_scales[idx] / init_scales[idx] > threshold:
                    mpm.x[idx] = m + threshold * init_scales[idx] * v
                elif init_scales[idx] / new_scales[idx] > threshold:
                    mpm.x[idx] = m + 1.0 / threshold * init_scales[idx] * v

    detF_buf = ti.ndarray(dtype=ti.f32, shape=n)

    @ti.kernel
    def read_detF(count: ti.i32, out: ti.types.ndarray()):
        for i in range(count):
            out[i] = mpm.F[i].determinant()

    out_dir = out_root / scene_id
    tri_dir = out_dir / "_native_triangles"
    tri_dir.mkdir(parents=True, exist_ok=True)

    body_rows = []

    def record(frame: int):
        info = mpm.particle_info()
        x = np.asarray(info["position"][:n], np.float64)
        if scene_id == "a6_freefall_v1":
            v = np.asarray(info["velocity"][:n], np.float64)
            read_detF(n, detF_buf)
            detF = detF_buf.to_numpy()[:n]
            body_rows.append({
                "frame": frame, "time": frame * cfg["frame_dt"],
                "com_x": x[:, 0].mean(), "com_y": x[:, 1].mean(),
                "com_z": x[:, 2].mean(),
                "com_vx": v[:, 0].mean(), "com_vy": v[:, 1].mean(),
                "com_vz": v[:, 2].mean(),
                "bbox_volume": float(np.prod(np.ptp(x, axis=0))),
                "detF_nonpositive_fraction": float((detF <= 0.0).mean()),
                "nonfinite_fraction": float(1.0 - np.isfinite(x).mean()),
                "particle_count": n,
            })
        else:
            exported = scaler.inverse(x)
            torch.save(torch.from_numpy(exported), tri_dir / f"{frame:04d}.pt")

    record(0)
    init_scales.from_numpy(
        calc_scales(np.asarray(mpm.particle_info()["position"][:n],
                               np.float32)))
    for frame in range(1, cfg["frames"] + 1):
        mpm.step(cfg["frame_dt"])
        new_scales.from_numpy(
            calc_scales(np.asarray(mpm.particle_info()["position"][:n],
                                   np.float32)))
        modify_positions()
        record(frame)
        if frame % max(1, cfg["frames"] // 5) == 0:
            print(f"[gasp-bench] {scene_id}: {frame}/{cfg['frames']}",
                  flush=True)

    if scene_id == "a6_freefall_v1":
        import pandas as pd
        pd.DataFrame(body_rows).to_csv(out_dir / "body_metrics.csv",
                                       index=False)
    else:
        subprocess.run([
            sys.executable,
            str(GAUSSIANBENCH / "adapters" / "gasp_triangle_export.py"),
            "--triangles-dir", str(tri_dir), "--output-dir", str(out_dir),
            "--system-name", "GASP", "--intervention", fault,
        ], check=True)

    def commit(repo):
        try:
            c = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo),
                               capture_output=True, text=True).stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=str(repo), capture_output=True, text=True).stdout.strip()
            return c + ("-dirty" if dirty else "")
        except Exception:
            return "unknown"

    import taichi as _ti
    meta = {
        "system_name": "GASP",
        "system_version": commit(REPO),
        "spec_version": "1.2.0",
        "representation": "GASP triangle soup (released teddy pseudomesh, "
                          f"{n // 3} stable triangles)",
        "wall_clock_seconds": round(time.perf_counter() - started, 3),
        "solver": f"released taichi_elements MPMSolver 64^3, taichi "
                  f"{_ti.__version__}, python {sys.version.split()[0]}",
        "intervention": fault,
        "stable_triangle_order": True,
        "notes": (
            f"Released GASP Taichi pipeline: native teddy pseudomesh vertices "
            f"simulated by released taichi_elements MPMSolver "
            f"(material={cfg['material']}, E_scale={cfg['E']}, res 64^3, "
            f"frame step {cfg['frame_dt']} s x {cfg['frames']} frames), with "
            f"GASP's released per-frame triangle size-control rule "
            f"(threshold=1.0) applied after every step exactly as in "
            f"taichi_examples/demo/3d.py. Placement: released Rescale "
            f"scale={cfg['scale']}, offset={cfg['offset']}; exports use the "
            f"released Rescale.inverse into the same frame as the frozen C5 "
            f"submission. Gravity={cfg['gravity'] if cfg['gravity'] is not None else 'released default (0,-9.8,0)'}"
            + (f"; uniform native initial velocity {velocity} chosen as "
               f"D^-1 d so the exported COM direction equals the scene's "
               f"declared direction" if velocity is not None else "")
            + ("; A6 body metrics are in native solver units (meters), "
               "computed from released solver state including per-particle "
               "F read from the solver's own field" if scene_id ==
               "a6_freefall_v1" else "")
            + (f"; CONTROLLED FAULT: {fault}" if fault != "none" else "")),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2),
                                       encoding="utf-8")
    print(f"[gasp-bench] wrote {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fault", default="none")
    args = ap.parse_args()
    run(args.scene, args.out.resolve(), args.fault)
