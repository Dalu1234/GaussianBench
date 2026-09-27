"""DEPRECATED: superseded by the shared bench harness.

The production Physics3D broad entry runs through
``PhysGaussian/bench/run_bench_scene.py`` with
``GAUSSIANBENCH_METHOD_ROOT`` pointed at this repository and
``GAUSSIANBENCH_SYSTEM_NAME=Physics3D`` (see
gaussianbench/adapters/README.md). That harness imposes boundary conditions
at both the grid and particle levels; the particle-only prescriptions in this
file are mass-diluted by the B-spline transfer stencil and do not transmit
the commanded deformation (the scorers correctly flag the result INVALID).
This file is retained only as diagnostic history of that finding and of the
E/1e7 native-unit mapping.
"""

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
MPM = REPO / "mpm_solver_warp"
sys.path.insert(0, str(MPM))
from mpm_solver_warp import MPM_Simulator_WARP  # noqa: E402
from warp_utils import MPMStateStruct  # noqa: E402

DEVICE = "cuda:0"
SUPPORTED = {
    "a1a_conservation_inplace_v1", "a2_beam_frequency_v1", "a3_wavespeed_v1",
    "a5_stiffness_v1", "c1_covariance_v1", "c2_blobhealth_shear_v1",
    "c4_captured_frequency_v1",
}


def body_volume(scene: dict) -> float:
    g = scene["geometry"]
    if g["type"] == "sphere":
        return 4.0 * math.pi * float(g["radius_m"]) ** 3 / 3.0
    if "size_m" in g:
        return float(np.prod(g["size_m"]))
    return float(np.prod(g["approx_extent_m"]))


def initial_velocity(scene: dict, rest: np.ndarray) -> np.ndarray:
    ic = scene.get("initial_conditions", {})
    velocity = np.broadcast_to(np.asarray(ic.get("linear_velocity_m_s", [0, 0, 0]), float), rest.shape).copy()
    omega = np.asarray(ic.get("angular_velocity_rad_s", [0, 0, 0]), float)
    velocity += np.cross(omega[None], rest - rest.mean(0))
    if ic.get("type") == "modal_velocity_pluck":
        b = 1.875104068711961
        sigma = (np.cosh(b) - np.cos(b)) / (np.sinh(b) + np.sin(b))
        s = (rest[:, 0] - rest[:, 0].min()) / np.ptp(rest[:, 0])
        phi = np.cosh(b*s)-np.cos(b*s)-sigma*(np.sinh(b*s)-np.sin(b*s))
        phi1 = np.cosh(b)-np.cos(b)-sigma*(np.sinh(b)-np.sin(b))
        velocity += (float(ic["tip_velocity_m_s"])*phi/phi1)[:, None] * np.asarray(ic["direction"], float)
    return velocity.astype(np.float32)


def initial_C(scene: dict, count: int) -> np.ndarray:
    C = np.zeros((count, 3, 3), np.float32)
    w = np.asarray(scene.get("initial_conditions", {}).get("angular_velocity_rad_s", [0, 0, 0]), float)
    C[:, 0, 1], C[:, 0, 2] = -w[2], w[1]
    C[:, 1, 0], C[:, 1, 2] = w[2], -w[0]
    C[:, 2, 0], C[:, 2, 1] = -w[1], w[0]
    return C


def region(expr: str, x: np.ndarray) -> np.ndarray:
    axis_name, op, value = expr.split()
    values = x[:, {"x": 0, "y": 1, "z": 2}[axis_name]]
    return values < float(value) if op == "<" else values > float(value)


# Scene boundary conditions are realized as kinematic driver particles: a
# masked boundary layer whose position and velocity are reset to the scene's
# commanded trajectory at the start of every substep, through the solver's
# released ``pre_p2g_operations`` hook (the same pre-p2g particle-state hook
# the release uses for impulses; its velocity-only siblings are
# ``enforce_particle_velocity_translation``/``_rotation``).  The driver layer
# is the boundary condition itself; everything else (transfers, grid update,
# stress, deformation-gradient integration) runs unmodified released kernels.
# Velocity-only enforcement was tried first and is too weak here: with one- to
# two-spacing masks against a dx = 2 x spacing grid, mass-weighted transfers
# dilute the constraint and the commanded displacement is not realized
# (position drift), which the scorers correctly flag as INVALID.
#
# Trajectory profiles (mode), with x0 the placed rest position:
#   0: x = x0, v = vel[p] (= 0 for clamps), for all time
#   1: x = x0 + min(t, t1) * vel[p]; v = vel[p] while t < t1, then 0 (ramp+hold)
#   2: x = x0 + (t1/pi) sin(pi t/t1) vel[p]; v = vel[p] cos(pi t/t1) while
#      t <= t1, then x = x0, v = 0 (half-sine displacement pulse)
#   3: x = x0 + t*vel while t < t1; x = x0 + (t2 - t)*vel while t < t2;
#      then x = x0, v = 0 (triangle shear ramp up/down)
@wp.struct
class PrescribedMotion:
    mask: wp.array(dtype=int)
    pos0: wp.array(dtype=wp.vec3)
    vel: wp.array(dtype=wp.vec3)
    mode: int
    t1: float
    t2: float


@wp.kernel
def apply_prescribed_motion(time: float, dt: float, state: MPMStateStruct, params: PrescribedMotion):
    p = wp.tid()
    if params.mask[p] == 1:
        if params.mode == 0:
            state.particle_x[p] = params.pos0[p]
            state.particle_v[p] = params.vel[p]
        elif params.mode == 1:
            if time < params.t1:
                state.particle_x[p] = params.pos0[p] + time * params.vel[p]
                state.particle_v[p] = params.vel[p]
            else:
                state.particle_x[p] = params.pos0[p] + params.t1 * params.vel[p]
                state.particle_v[p] = wp.vec3(0.0, 0.0, 0.0)
        elif params.mode == 2:
            if time <= params.t1:
                s = 3.141592653589793 * time / params.t1
                state.particle_x[p] = params.pos0[p] + (params.t1 / 3.141592653589793) * wp.sin(s) * params.vel[p]
                state.particle_v[p] = params.vel[p] * wp.cos(s)
            else:
                state.particle_x[p] = params.pos0[p]
                state.particle_v[p] = wp.vec3(0.0, 0.0, 0.0)
        elif params.mode == 3:
            if time < params.t1:
                state.particle_x[p] = params.pos0[p] + time * params.vel[p]
                state.particle_v[p] = params.vel[p]
            elif time < params.t2:
                state.particle_x[p] = params.pos0[p] + (params.t2 - time) * params.vel[p]
                state.particle_v[p] = -params.vel[p]
            else:
                state.particle_x[p] = params.pos0[p]
                state.particle_v[p] = wp.vec3(0.0, 0.0, 0.0)


def make_prescribed(scene: dict, rest: np.ndarray, offset: np.ndarray) -> list:
    """Build PrescribedMotion params (mask, rest position, velocity, profile)."""
    count, result = len(rest), []
    placed = rest + offset

    def add(mask: np.ndarray, vel: np.ndarray, mode: int, t1: float = 0.0, t2: float = 0.0):
        params = PrescribedMotion()
        params.mask = wp.from_numpy(mask.astype(np.int32), dtype=int, device="cuda:0")
        params.pos0 = wp.from_numpy(placed.astype(np.float32), dtype=wp.vec3, device="cuda:0")
        params.vel = wp.from_numpy(vel.astype(np.float32), dtype=wp.vec3, device="cuda:0")
        params.mode, params.t1, params.t2 = mode, t1, t2
        result.append(params)

    for bc in (scene.get("boundary_conditions") or {}).values():
        if not isinstance(bc, dict) or "region" not in bc:
            continue
        mask = region(bc["region"], rest)
        if bc.get("type") == "fixed":
            add(mask, np.zeros((count, 3)), 0)
        elif bc.get("type") == "displacement_ramp":
            d, end = np.asarray(bc["displacement_m"], float), float(bc["ramp_end_s"])
            add(mask, np.broadcast_to(d / end, (count, 3)).copy(), 1, end)
        elif bc.get("type") == "prescribed_displacement_pulse":
            a = float(bc["amplitude_m"]) * np.asarray(bc["direction"], float)
            width = float(bc["width_s"])
            add(mask, np.broadcast_to(a * np.pi / width, (count, 3)).copy(), 2, width)
    affine = (scene.get("boundary_conditions") or {}).get("all_faces")
    if isinstance(affine, dict) and affine.get("type") == "displacement_ramp_affine":
        spacing, lo, hi = float(scene["geometry"]["particle_spacing_m"]), rest.min(0), rest.max(0)
        mask = np.any((rest < lo + 2.1 * spacing) | (rest > hi - 2.1 * spacing), axis=1)
        A = np.asarray(affine["F_target"], float) - np.eye(3)
        about, end = np.asarray(affine["about"], float), float(affine["ramp_end_s"])
        add(mask, (rest - about) @ (A / end).T, 1, end)
    shear = (scene.get("boundary_conditions") or {}).get("shear_drive")
    if isinstance(shear, dict):
        spacing, lo, hi = float(scene["geometry"]["particle_spacing_m"]), rest[:, 1].min(), rest[:, 1].max()
        mask = (rest[:, 1] < lo + 1.6 * spacing) | (rest[:, 1] > hi - 1.6 * spacing)
        peak, duration = float(shear["gamma_peak"]), float(scene["simulation"]["duration_s"])
        vel = np.zeros((count, 3))
        vel[:, 0] = 2 * peak / duration * rest[:, 1]
        add(mask, vel, 3, duration / 2, duration)
    return result


def run(scene_path: Path, out: Path) -> None:
    scene, started = json.loads(scene_path.read_text()), time.perf_counter()
    sid = scene["scene_id"]
    if sid not in SUPPORTED:
        raise SystemExit("NOT SUBMITTED: scene is outside Physics3D's released nonthermal MPM contract")
    rest = np.load(scene_path.parent / scene["geometry"]["particle_positions_file"]).astype(np.float64)
    extent = np.ptp(rest, axis=0)
    grid_lim = 1.2 if extent.max() > 0.9 else 1.0
    offset = np.full(3, grid_lim/2)-0.5*(rest.min(0)+rest.max(0))
    placed, spacing = rest+offset, float(scene["geometry"]["particle_spacing_m"])
    grid = max(24, int(math.ceil(grid_lim/(2*spacing))))
    material = scene["material"]
    E, nu, density = float(material["youngs_modulus_pa"]), float(material["poisson_ratio"]), float(material["density_kg_m3"])
    wave = math.sqrt(E*(1-nu)/((1+nu)*(1-2*nu))/density)
    fps, duration = float(scene["simulation"]["output_fps"]), float(scene["simulation"]["duration_s"])
    frame_dt, dx = 1/fps, grid_lim/grid
    substeps = max(1, math.ceil(frame_dt/(0.25*dx/wave)))
    dt, count, pvol = frame_dt/substeps, len(rest), body_volume(scene)/len(rest)
    tier_c = sid in {"c1_covariance_v1", "c2_blobhealth_shear_v1"}
    variance = (0.5*spacing)**2
    cov = torch.as_tensor(np.tile([variance, 0, 0, variance, 0, variance], (count, 1)), device="cuda", dtype=torch.float32) if tier_c else None
    solver = MPM_Simulator_WARP(count, n_grid=grid, grid_lim=grid_lim, device=DEVICE)
    solver.load_initial_data_from_torch(torch.as_tensor(placed, device="cuda", dtype=torch.float32), torch.full((count,), pvol, device="cuda"), cov, n_grid=grid, grid_lim=grid_lim, device=DEVICE)
    solver.import_particle_v_from_torch(torch.as_tensor(initial_velocity(scene, rest), device="cuda"), device=DEVICE)
    solver.import_particle_C_from_torch(torch.as_tensor(initial_C(scene, count), device="cuda"), device=DEVICE)
    solver.import_particle_F_from_torch(torch.eye(3, device="cuda").repeat(count, 1, 1), device=DEVICE)
    # Physics3D's released ``jelly_base`` is its pure corotated branch.  The
    # default ``jelly`` branch additionally expects learned Maxwell fields.
    # Physics3D stores Young's modulus in units of 1e7 Pa: its released
    # compute_mu_lam_from_E_nu multiplies E by 1e7 (native ball config uses
    # E=0.2 for 2 MPa). The frozen scenes give E in Pa, so convert here.
    solver.set_parameters_dict({"material": "jelly_base", "E": E / 1.0e7, "nu": nu, "g": scene["simulation"]["gravity"], "density": density, "grid_v_damping_scale": 1.0}, device=DEVICE)
    solver.finalize_mu_lam(device=DEVICE)
    if tier_c:
        solver.mpm_model.update_cov_with_F = True
        solver.mpm_state.particle_cov = solver.mpm_state.particle_init_cov
    for params in make_prescribed(scene, rest, offset):
        solver.pre_p2g_operations.append(apply_prescribed_motion)
        solver.impulse_params.append(params)
    trajectories, covariances = [], []
    mass = density*pvol
    def record(frame: int):
        x = solver.export_particle_x_to_torch().detach().cpu().numpy()-offset
        v = solver.export_particle_v_to_torch().detach().cpu().numpy()
        F = solver.export_particle_F_to_torch().detach().cpu().numpy().reshape(-1, 3, 3)
        data = {"frame": frame, "time": frame/fps, "particle_id": np.arange(count), "x": x[:,0], "y": x[:,1], "z": x[:,2], "vx": v[:,0], "vy": v[:,1], "vz": v[:,2], "mass": mass}
        for i in range(3):
            for j in range(3): data[f"F{i}{j}"] = F[:,i,j]
        trajectories.append(pd.DataFrame(data))
        if tier_c:
            c = solver.export_particle_cov_to_torch(DEVICE).detach().cpu().numpy().reshape(-1, 6)
            covariances.append(pd.DataFrame({"frame": frame, "particle_id": np.arange(count), "sxx": c[:,0], "sxy": c[:,1], "sxz": c[:,2], "syy": c[:,3], "syz": c[:,4], "szz": c[:,5]}))
    record(0)
    frames = round(duration*fps)
    for frame in range(1, frames+1):
        for _ in range(substeps):
            solver.p2g2p(frame, dt, device=DEVICE)
        record(frame)
        if frame % max(1, frames//10) == 0: print(f"{sid}: {frame}/{frames}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(trajectories, ignore_index=True).to_parquet(out/"trajectory.parquet", index=False)
    if covariances: pd.concat(covariances, ignore_index=True).to_parquet(out/"covariances.parquet", index=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    meta = {"system_name": "Physics3D", "system_version": commit, "spec_version": "1.2.0", "columns_present": ["vx","vy","vz","mass"]+[f"F{i}{j}" for i in range(3) for j in range(3)], "wall_clock_seconds": round(time.perf_counter()-started, 3), "hardware": torch.cuda.get_device_name(), "notes": f"Released Physics3D Warp MPM (jelly_base corotated branch); fixed grid {grid}^3; dt={dt:.9g}; {substeps} substeps/frame; translation-only placement inverted on export; native covariance state for Tier C. Parameter mapping: scene E [Pa] passed to the solver as E/1e7 because Physics3D's released compute_mu_lam_from_E_nu stores E in units of 1e7 Pa (native ball config E=0.2). Boundary conditions realized as kinematic driver particles: masked boundary layers have position and velocity reset to the scene-commanded trajectory each substep through the released pre-p2g pre_p2g_operations hook; the scored interior evolves entirely through released kernels."}
    (out/"meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--scene", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); wp.init(); wp.config.verify_cuda = True; run(args.scene.resolve(), args.out.resolve())
