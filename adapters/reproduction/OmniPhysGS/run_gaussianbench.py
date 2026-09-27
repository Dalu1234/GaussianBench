"""Run frozen GaussianBench scenes through OmniPhysGS's native PyTorch MPM.

This adapter changes no solver equations. It maps scene coordinates into the
released fixed [0,1]^3 grid by translation only, supplies the frozen material
parameters to the released constitutive laws, and serializes x, v, F, and
the covariance push-forward used by OmniPhysGS rendering.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import warp as wp

REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO))

from src.constitutive_models.physical_constitutive_models.elasticity import (  # noqa: E402
    CorotatedElasticity, StVKElasticity)
from src.mpm_core.mpm_model import MPMModel  # noqa: E402


def parse_region(expr: str, rest: np.ndarray) -> np.ndarray:
    axis_name, op, value = expr.split()
    axis = {"x": 0, "y": 1, "z": 2}[axis_name]
    return (rest[:, axis] < float(value) if op == "<"
            else rest[:, axis] > float(value))


def scene_volume(scene: dict) -> float:
    g = scene["geometry"]
    if g["type"] == "sphere":
        return 4.0 * math.pi * float(g["radius_m"]) ** 3 / 3.0
    if "size_m" in g:
        return float(np.prod(g["size_m"]))
    extent = np.asarray(g.get("approx_extent_m"), float)
    return float(np.prod(extent))


def initial_velocity(scene: dict, rest: np.ndarray) -> np.ndarray:
    ic = scene.get("initial_conditions", {})
    v = np.zeros_like(rest)
    if "linear_velocity_m_s" in ic:
        v += np.asarray(ic["linear_velocity_m_s"], float)
    if "angular_velocity_rad_s" in ic:
        omega = np.asarray(ic["angular_velocity_rad_s"], float)
        v += np.cross(omega[None], rest - rest.mean(axis=0))
    if ic.get("type") == "modal_velocity_pluck":
        b = 1.875104068711961
        sigma = (np.cosh(b) - np.cos(b)) / (np.sinh(b) + np.sin(b))
        s = (rest[:, 0] - rest[:, 0].min()) / np.ptp(rest[:, 0])
        phi = (np.cosh(b*s) - np.cos(b*s)
               - sigma * (np.sinh(b*s) - np.sin(b*s)))
        phi_tip = np.cosh(b) - np.cos(b) - sigma*(np.sinh(b) - np.sin(b))
        v += (float(ic["tip_velocity_m_s"]) * phi / phi_tip)[:, None] * \
            np.asarray(ic["direction"], float)[None]
    return v


def initial_C(scene: dict, n: int) -> np.ndarray:
    C = np.zeros((n, 3, 3), dtype=np.float32)
    omega = np.asarray(
        scene.get("initial_conditions", {}).get(
            "angular_velocity_rad_s", [0, 0, 0]), float)
    C[:, 0, 1], C[:, 0, 2] = -omega[2], omega[1]
    C[:, 1, 0], C[:, 1, 2] = omega[2], -omega[0]
    C[:, 2, 0], C[:, 2, 1] = -omega[1], omega[0]
    return C


class Writer:
    def __init__(self, out: Path, scene_path: Path, rest_cov: np.ndarray | None,
                 offset: np.ndarray):
        self.out, self.scene_path = out, scene_path
        self.rest_cov, self.offset = rest_cov, offset
        self.rows, self.cov_rows = [], []
        self.started = time.perf_counter()

    def add(self, frame: int, t: float, x, v, F, mass: float):
        x = x.detach().cpu().numpy().astype(np.float32) - self.offset
        v = v.detach().cpu().numpy().astype(np.float32)
        fg = F.detach().cpu().numpy().astype(np.float32)
        n = len(x)
        data = {
            "frame": np.full(n, frame), "time": np.full(n, t),
            "particle_id": np.arange(n), "x": x[:, 0], "y": x[:, 1],
            "z": x[:, 2], "vx": v[:, 0], "vy": v[:, 1], "vz": v[:, 2],
            "mass": np.full(n, mass, np.float32),
        }
        for i in range(3):
            for j in range(3):
                data[f"F{i}{j}"] = fg[:, i, j]
        self.rows.append(pd.DataFrame(data))
        if self.rest_cov is not None:
            cov = np.einsum("nij,njk,nlk->nil", fg, self.rest_cov, fg)
            self.cov_rows.append(pd.DataFrame({
                "frame": np.full(n, frame), "particle_id": np.arange(n),
                "sxx": cov[:, 0, 0], "sxy": cov[:, 0, 1],
                "sxz": cov[:, 0, 2], "syy": cov[:, 1, 1],
                "syz": cov[:, 1, 2], "szz": cov[:, 2, 2],
            }))

    def finish(self, notes: list[str]):
        self.out.mkdir(parents=True, exist_ok=True)
        pd.concat(self.rows, ignore_index=True).to_parquet(
            self.out / "trajectory.parquet", index=False)
        if self.cov_rows:
            pd.concat(self.cov_rows, ignore_index=True).to_parquet(
                self.out / "covariances.parquet", index=False)
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        except Exception:
            commit = "unknown"
        (self.out / "meta.json").write_text(json.dumps({
            "system_name": "OmniPhysGS",
            "system_version": commit,
            "spec_version": "1.2.0",
            "columns_present": ["vx", "vy", "vz", "mass"] +
                [f"F{i}{j}" for i in range(3) for j in range(3)],
            "wall_clock_seconds": round(time.perf_counter()-self.started, 3),
            "hardware": torch.cuda.get_device_name(),
            "notes": " | ".join(notes),
        }, indent=2), encoding="utf-8")


class Prescription:
    def __init__(self, mask, state):
        self.mask = torch.as_tensor(mask, device="cuda", dtype=torch.bool)
        self.state = state

    def apply(self, x, v, t):
        target_x, target_v = self.state(t)
        x[self.mask] = target_x[self.mask]
        v[self.mask] = target_v[self.mask]


def prescriptions(scene, rest, offset):
    n = len(rest)
    base = rest + offset
    result = []
    for bc in (scene.get("boundary_conditions") or {}).values():
        if not isinstance(bc, dict):
            continue
        kind = bc.get("type")
        if kind == "fixed":
            mask = parse_region(bc["region"], rest)
            def fixed(t, base=base):
                return (torch.as_tensor(base, device="cuda", dtype=torch.float32),
                        torch.zeros((n, 3), device="cuda"))
            result.append(Prescription(mask, fixed))
        elif kind == "displacement_ramp":
            mask = parse_region(bc["region"], rest)
            disp = np.asarray(bc["displacement_m"], float)
            T = float(bc["ramp_end_s"])
            def ramp(t, base=base, disp=disp, T=T):
                alpha = min(t/T, 1.0)
                px = base + alpha*disp
                pv = np.zeros_like(base)
                if t < T:
                    pv[:] = disp/T
                return (torch.as_tensor(px, device="cuda", dtype=torch.float32),
                        torch.as_tensor(pv, device="cuda", dtype=torch.float32))
            result.append(Prescription(mask, ramp))
        elif kind == "prescribed_displacement_pulse":
            mask = parse_region(bc["region"], rest)
            amp = float(bc["amplitude_m"]) * np.asarray(bc["direction"], float)
            W = float(bc["width_s"])
            def pulse(t, base=base, amp=amp, W=W):
                if 0 <= t <= W:
                    px = base + amp*np.sin(np.pi*t/W)
                    pv = np.broadcast_to(
                        amp*(np.pi/W)*np.cos(np.pi*t/W), base.shape).copy()
                else:
                    px, pv = base, np.zeros_like(base)
                return (torch.as_tensor(px, device="cuda", dtype=torch.float32),
                        torch.as_tensor(pv, device="cuda", dtype=torch.float32))
            result.append(Prescription(mask, pulse))

    all_faces = (scene.get("boundary_conditions") or {}).get("all_faces")
    if isinstance(all_faces, dict) and all_faces.get("type") == \
            "displacement_ramp_affine":
        spacing = float(scene["geometry"]["particle_spacing_m"])
        lo, hi = rest.min(0), rest.max(0)
        mask = np.any((rest < lo+2.1*spacing) | (rest > hi-2.1*spacing), axis=1)
        Ft = np.asarray(all_faces["F_target"], float)
        A = Ft - np.eye(3)
        about = np.asarray(all_faces["about"], float)
        T = float(all_faces["ramp_end_s"])
        def affine(t, rest=rest, offset=offset, A=A, about=about, T=T):
            alpha = min(t/T, 1.0)
            px = about + (rest-about) @ (np.eye(3)+alpha*A).T + offset
            pv = (rest-about) @ (A/T).T if t < T else np.zeros_like(rest)
            return (torch.as_tensor(px, device="cuda", dtype=torch.float32),
                    torch.as_tensor(pv, device="cuda", dtype=torch.float32))
        result.append(Prescription(mask, affine))

    shear = (scene.get("boundary_conditions") or {}).get("shear_drive")
    if isinstance(shear, dict):
        spacing = float(scene["geometry"]["particle_spacing_m"])
        lo, hi = rest.min(0), rest.max(0)
        mask = (rest[:, 1] < lo[1]+1.6*spacing) | \
               (rest[:, 1] > hi[1]-1.6*spacing)
        peak, T = float(shear["gamma_peak"]), float(scene["simulation"]["duration_s"])
        def shear_state(t, rest=rest, offset=offset, peak=peak, T=T):
            if t < T/2:
                gamma, rate = 2*peak*t/T, 2*peak/T
            elif t < T:
                gamma, rate = 2*peak*(T-t)/T, -2*peak/T
            else:
                gamma, rate = 0.0, 0.0
            px = rest.copy()
            px[:, 0] += gamma*rest[:, 1]
            px += offset
            pv = np.zeros_like(rest)
            pv[:, 0] = rate*rest[:, 1]
            return (torch.as_tensor(px, device="cuda", dtype=torch.float32),
                    torch.as_tensor(pv, device="cuda", dtype=torch.float32))
        result.append(Prescription(mask, shear_state))
    return result


def register_grid_bcs(sim, scene, rest, offset):
    """Apply scene Dirichlet data at OmniPhysGS's native post-grid hook."""
    grid_pos = sim.grid_x * sim.dx
    lo, hi = rest.min(0) + offset, rest.max(0) + offset
    pad = 1.1 * sim.dx
    body_cross_section = (
        (grid_pos[:, 1] >= lo[1]-pad) & (grid_pos[:, 1] <= hi[1]+pad) &
        (grid_pos[:, 2] >= lo[2]-pad) & (grid_pos[:, 2] <= hi[2]+pad))

    for bc in (scene.get("boundary_conditions") or {}).values():
        if not isinstance(bc, dict) or "region" not in bc:
            continue
        kind = bc.get("type")
        axis_name, op, value = bc["region"].split()
        axis = {"x": 0, "y": 1, "z": 2}[axis_name]
        plane0 = float(value) + offset[axis]
        if kind == "fixed":
            if op == "<":
                target = body_cross_section & (grid_pos[:, axis] < plane0+pad)
            else:
                target = body_cross_section & (grid_pos[:, axis] > plane0-pad)
            def fixed_grid(model, target=target):
                model.grid_mv[target] = 0.0
            sim.post_grid_process.append(fixed_grid)
        elif kind == "displacement_ramp":
            disp = torch.as_tensor(
                bc["displacement_m"], device="cuda", dtype=torch.float32)
            T = float(bc["ramp_end_s"])
            end_plane = plane0 + float(disp[axis])
            lower, upper = min(plane0, end_plane)-pad, max(plane0, end_plane)+pad
            target = body_cross_section & (grid_pos[:, axis] >= lower) & \
                     (grid_pos[:, axis] <= upper)
            def ramp_grid(model, target=target, disp=disp, T=T):
                if model.time < T:
                    model.grid_mv[target] = disp/T
                else:
                    model.grid_mv[target] = 0.0
            sim.post_grid_process.append(ramp_grid)
        elif kind == "prescribed_displacement_pulse":
            amp = torch.as_tensor(
                float(bc["amplitude_m"])*np.asarray(bc["direction"], float),
                device="cuda", dtype=torch.float32)
            W = float(bc["width_s"])
            target = body_cross_section & (torch.abs(grid_pos[:, axis]-plane0) < pad)
            def pulse_grid(model, target=target, amp=amp, W=W):
                if 0 <= model.time <= W:
                    model.grid_mv[target] = amp*(math.pi/W)*math.cos(
                        math.pi*model.time/W)
                else:
                    model.grid_mv[target] = 0.0
            sim.post_grid_process.append(pulse_grid)


def run(scene_path: Path, out: Path):
    scene = json.loads(scene_path.read_text())
    sid = scene["scene_id"]
    unsupported = {
        "a1b_conservation_drifting_v1": "fixed [0,1]^3 released grid cannot contain 11.5 m drift",
        "a4_meltfront_v1": "no thermal/phase state",
        "c2_blobhealth_melt_v1": "no thermal/phase state",
        "c3_roundtrip_v1": "renderer round trip not ported",
        "a6_freefall_v1": "representation-native HDF5 contract is not this trajectory adapter",
        "c5_triangle_shape_transport_v1": "not a triangle-soup representation",
        "m1_material_validation_v1": "requires a stress-strain submission",
    }
    if sid in unsupported:
        raise SystemExit("NOT SUBMITTED: " + unsupported[sid])

    pos_file = scene["geometry"].get("particle_positions_file")
    if not pos_file:
        raise SystemExit("NOT SUBMITTED: no frozen particle positions")
    rest = np.load(scene_path.parent / pos_file).astype(np.float64)
    extent = np.ptp(rest, axis=0)
    if extent.max() >= 0.998:
        offset = np.array([0.5, 0.5, 0.5]) - 0.5*(rest.min(0)+rest.max(0))
    else:
        offset = np.array([0.5, 0.5, 0.5]) - 0.5*(rest.min(0)+rest.max(0))
    placed = rest + offset
    if placed.min() <= 0 or placed.max() >= 1:
        raise SystemExit("NOT SUBMITTED: frozen geometry does not fit released fixed grid")

    spacing = float(scene["geometry"]["particle_spacing_m"])
    ngrid = max(24, int(math.ceil(1.0/(2.0*spacing))))
    fps = float(scene["simulation"]["output_fps"])
    duration = float(scene["simulation"]["duration_s"])
    frame_dt = 1.0/fps
    mat = scene["material"]
    if "regions" in mat:
        E = np.where(rest[:, 0] < float(mat["interface_x_m"]),
                     float(mat["regions"][0]["youngs_modulus_pa"]),
                     float(mat["regions"][1]["youngs_modulus_pa"]))
        nu = np.where(rest[:, 0] < float(mat["interface_x_m"]),
                      float(mat["regions"][0]["poisson_ratio"]),
                      float(mat["regions"][1]["poisson_ratio"]))
        density = float(mat["regions"][0]["density_kg_m3"])
    else:
        E = np.full(len(rest), float(mat["youngs_modulus_pa"]))
        nu = np.full(len(rest), float(mat["poisson_ratio"]))
        density = float(mat["density_kg_m3"])
    c = np.sqrt((E.max()*(1-nu.min()) /
                 ((1+nu.min())*(1-2*nu.min())))/density)
    dt_cfl = 0.25*(1.0/ngrid)/c
    substeps = max(1, int(math.ceil(frame_dt/dt_cfl)))
    dt = frame_dt/substeps
    volume = scene_volume(scene)
    cube = volume ** (1/3)
    sim = MPMModel(
        {"num_grids": ngrid, "dt": dt,
         "gravity": scene["simulation"]["gravity"],
         "clip_bound": 0.1, "damping": 1.0},
        {"center": [0.5]*3, "size": [cube]*3, "rho": density},
        torch.as_tensor(placed, device="cuda", dtype=torch.float32),
        device="cuda")
    register_grid_bcs(sim, scene, rest, offset)

    x = torch.as_tensor(placed, device="cuda", dtype=torch.float32)
    v = torch.as_tensor(initial_velocity(scene, rest), device="cuda", dtype=torch.float32)
    C = torch.as_tensor(initial_C(scene, len(rest)), device="cuda")
    F = torch.eye(3, device="cuda").repeat(len(rest), 1, 1)
    logE = torch.as_tensor(np.log(E), device="cuda", dtype=torch.float32)
    nut = torch.as_tensor(nu, device="cuda", dtype=torch.float32)
    corotated = CorotatedElasticity().cuda()
    stvk = StVKElasticity().cuda()
    bc = prescriptions(scene, rest, offset)
    tier_c = sid in {"c1_covariance_v1", "c2_blobhealth_shear_v1"}
    rest_cov = None
    if tier_c:
        rest_cov = np.tile(
            np.eye(3)*(0.5*spacing)**2, (len(rest), 1, 1)).astype(np.float32)
    writer = Writer(out, scene_path, rest_cov, offset)
    mass = density*volume/len(rest)
    notes = [
        "Released OmniPhysGS MPMModel and constitutive modules.",
        f"Translation-only placement offset {offset.tolist()} was inverted on export.",
        f"Native fixed grid: {ngrid}^3, dt={dt:.9g}, {substeps} substeps/frame.",
        "Covariance uses the released OmniPhysGS F Sigma0 F^T rendering contract.",
    ]
    if sid == "b1_bimaterial_bar_v1":
        notes.append(
            "Soft region uses released CorotatedElasticity; stiff region uses "
            "released StVKElasticity, which shares the prescribed linear response "
            "in B1's one-percent-strain regime.")

    writer.add(0, 0.0, x, v, F, mass)
    nframes = int(round(duration*fps))
    with torch.no_grad():
        for frame in range(1, nframes+1):
            for _ in range(substeps):
                stress_c = corotated(F, logE, nut)
                if sid == "b1_bimaterial_bar_v1":
                    stress_s = stvk(F, logE, nut)
                    stiff = torch.as_tensor(
                        rest[:, 0] > float(mat["interface_x_m"]),
                        device="cuda")
                    stress = torch.where(stiff[:, None, None], stress_s, stress_c)
                else:
                    stress = stress_c
                x, v, C, F = sim(x, v, C, F, stress)
                for prescription in bc:
                    prescription.apply(x, v, sim.time)
            writer.add(frame, frame/fps, x, v, F, mass)
            if frame % max(1, nframes//10) == 0:
                print(f"{sid}: {frame}/{nframes}", flush=True)
    writer.finish(notes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    wp.init()
    run(args.scene.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
