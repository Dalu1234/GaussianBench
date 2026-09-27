"""Run frozen GaussianBench scenes with PhysDreamer's released Warp MPM core."""

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
MPM = REPO / "physdreamer" / "warp_mpm"
sys.path.insert(0, str(MPM))

from mpm_data_structure import (  # noqa: E402
    MPMModelStruct, MPMStateStruct, set_mat33_to_identity)
from mpm_solver_diff import MPMWARPDiff  # noqa: E402
from warp_utils import from_torch_safe  # noqa: E402

DEVICE = "cuda:0"


def region(expr: str, x: np.ndarray) -> np.ndarray:
    name, op, value = expr.split()
    axis = {"x": 0, "y": 1, "z": 2}[name]
    return x[:, axis] < float(value) if op == "<" else x[:, axis] > float(value)


def volume(scene: dict) -> float:
    geometry = scene["geometry"]
    if geometry["type"] == "sphere":
        return 4 * math.pi * float(geometry["radius_m"]) ** 3 / 3
    if "size_m" in geometry:
        return float(np.prod(geometry["size_m"]))
    return float(np.prod(geometry["approx_extent_m"]))


def initial_velocity(scene: dict, rest: np.ndarray) -> np.ndarray:
    ic = scene.get("initial_conditions", {})
    velocity = np.zeros_like(rest)
    velocity += np.asarray(ic.get("linear_velocity_m_s", [0, 0, 0]), float)
    if "angular_velocity_rad_s" in ic:
        omega = np.asarray(ic["angular_velocity_rad_s"], float)
        velocity += np.cross(omega[None], rest - rest.mean(0))
    if ic.get("type") == "modal_velocity_pluck":
        b = 1.875104068711961
        sigma = (np.cosh(b) - np.cos(b)) / (np.sinh(b) + np.sin(b))
        s = (rest[:, 0] - rest[:, 0].min()) / np.ptp(rest[:, 0])
        phi = np.cosh(b*s)-np.cos(b*s)-sigma*(np.sinh(b*s)-np.sin(b*s))
        phi1 = np.cosh(b)-np.cos(b)-sigma*(np.sinh(b)-np.sin(b))
        direction = np.asarray(ic["direction"], float)
        velocity += (float(ic["tip_velocity_m_s"])*phi/phi1)[:, None] * direction
    return velocity


def initial_C(scene: dict, count: int) -> np.ndarray:
    C = np.zeros((count, 3, 3), np.float32)
    omega = np.asarray(scene.get("initial_conditions", {}).get(
        "angular_velocity_rad_s", [0, 0, 0]), float)
    C[:, 0, 1], C[:, 0, 2] = -omega[2], omega[1]
    C[:, 1, 0], C[:, 1, 2] = omega[2], -omega[0]
    C[:, 2, 0], C[:, 2, 1] = -omega[1], omega[0]
    return C


class Boundary:
    def __init__(self, mask: np.ndarray, target):
        self.mask = torch.as_tensor(mask, device="cuda", dtype=torch.bool)
        self.target = target

    def apply(self, state, t: float):
        x = wp.to_torch(state.particle_x)
        v = wp.to_torch(state.particle_v)
        target_x, target_v = self.target(t)
        x[self.mask], v[self.mask] = target_x[self.mask], target_v[self.mask]


def boundaries(scene: dict, rest: np.ndarray, offset: np.ndarray) -> list[Boundary]:
    result, base, count = [], rest + offset, len(rest)
    for bc in (scene.get("boundary_conditions") or {}).values():
        if not isinstance(bc, dict) or "region" not in bc:
            continue
        mask = region(bc["region"], rest)
        if bc.get("type") == "fixed":
            def fixed(t, base=base):
                return (torch.as_tensor(base, device="cuda", dtype=torch.float32),
                        torch.zeros((count, 3), device="cuda"))
            result.append(Boundary(mask, fixed))
        elif bc.get("type") == "displacement_ramp":
            displacement = np.asarray(bc["displacement_m"], float)
            end = float(bc["ramp_end_s"])
            def ramp(t, base=base, displacement=displacement, end=end):
                alpha = min(t/end, 1.0)
                position = base + alpha*displacement
                velocity = np.broadcast_to(displacement/end, base.shape).copy() if t < end else np.zeros_like(base)
                return (torch.as_tensor(position, device="cuda", dtype=torch.float32),
                        torch.as_tensor(velocity, device="cuda", dtype=torch.float32))
            result.append(Boundary(mask, ramp))
        elif bc.get("type") == "prescribed_displacement_pulse":
            amplitude = float(bc["amplitude_m"])*np.asarray(bc["direction"], float)
            width = float(bc["width_s"])
            def pulse(t, base=base, amplitude=amplitude, width=width):
                if 0 <= t <= width:
                    position = base + amplitude*np.sin(np.pi*t/width)
                    velocity = np.broadcast_to(amplitude*np.pi/width*np.cos(np.pi*t/width), base.shape).copy()
                else:
                    position, velocity = base, np.zeros_like(base)
                return (torch.as_tensor(position, device="cuda", dtype=torch.float32),
                        torch.as_tensor(velocity, device="cuda", dtype=torch.float32))
            result.append(Boundary(mask, pulse))

    affine = (scene.get("boundary_conditions") or {}).get("all_faces")
    if isinstance(affine, dict) and affine.get("type") == "displacement_ramp_affine":
        spacing = float(scene["geometry"]["particle_spacing_m"])
        lo, hi = rest.min(0), rest.max(0)
        mask = np.any((rest < lo+2.1*spacing) | (rest > hi-2.1*spacing), axis=1)
        A = np.asarray(affine["F_target"], float) - np.eye(3)
        about, end = np.asarray(affine["about"], float), float(affine["ramp_end_s"])
        def affine_target(t, rest=rest, offset=offset, A=A, about=about, end=end):
            alpha = min(t/end, 1.0)
            position = about + (rest-about) @ (np.eye(3)+alpha*A).T + offset
            velocity = (rest-about) @ (A/end).T if t < end else np.zeros_like(rest)
            return (torch.as_tensor(position, device="cuda", dtype=torch.float32),
                    torch.as_tensor(velocity, device="cuda", dtype=torch.float32))
        result.append(Boundary(mask, affine_target))

    shear = (scene.get("boundary_conditions") or {}).get("shear_drive")
    if isinstance(shear, dict):
        spacing = float(scene["geometry"]["particle_spacing_m"])
        lo, hi = rest[:, 1].min(), rest[:, 1].max()
        mask = (rest[:, 1] < lo+1.6*spacing) | (rest[:, 1] > hi-1.6*spacing)
        peak, duration = float(shear["gamma_peak"]), float(scene["simulation"]["duration_s"])
        def shear_target(t, rest=rest, offset=offset, peak=peak, duration=duration):
            if t < duration/2:
                gamma, rate = 2*peak*t/duration, 2*peak/duration
            elif t < duration:
                gamma, rate = 2*peak*(duration-t)/duration, -2*peak/duration
            else:
                gamma, rate = 0.0, 0.0
            position, velocity = rest.copy(), np.zeros_like(rest)
            position[:, 0] += gamma*rest[:, 1]
            position += offset
            velocity[:, 0] = rate*rest[:, 1]
            return (torch.as_tensor(position, device="cuda", dtype=torch.float32),
                    torch.as_tensor(velocity, device="cuda", dtype=torch.float32))
        result.append(Boundary(mask, shear_target))
    return result


class Writer:
    def __init__(self, out: Path, offset: np.ndarray, native_covariance: bool):
        self.out, self.offset = out, offset
        self.native_covariance = native_covariance
        self.trajectory, self.covariance = [], []
        self.started = time.perf_counter()

    def add(self, frame: int, t: float, state, mass: float):
        x = wp.to_torch(state.particle_x).detach().cpu().numpy() - self.offset
        v = wp.to_torch(state.particle_v).detach().cpu().numpy()
        F = wp.to_torch(state.particle_F).detach().cpu().numpy()
        count = len(x)
        data = {"frame": frame, "time": t, "particle_id": np.arange(count),
                "x": x[:, 0], "y": x[:, 1], "z": x[:, 2],
                "vx": v[:, 0], "vy": v[:, 1], "vz": v[:, 2], "mass": mass}
        for i in range(3):
            for j in range(3):
                data[f"F{i}{j}"] = F[:, i, j]
        self.trajectory.append(pd.DataFrame(data))
        if self.native_covariance:
            cov = wp.to_torch(state.particle_cov).detach().cpu().numpy().reshape(-1, 6)
            self.covariance.append(pd.DataFrame({"frame": frame, "particle_id": np.arange(count),
                "sxx": cov[:, 0], "sxy": cov[:, 1], "sxz": cov[:, 2],
                "syy": cov[:, 3], "syz": cov[:, 4], "szz": cov[:, 5]}))

    def finish(self, notes: list[str]):
        self.out.mkdir(parents=True, exist_ok=True)
        pd.concat(self.trajectory, ignore_index=True).to_parquet(self.out/"trajectory.parquet", index=False)
        if self.covariance:
            pd.concat(self.covariance, ignore_index=True).to_parquet(self.out/"covariances.parquet", index=False)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        meta = {"system_name": "PhysDreamer", "system_version": commit,
                "spec_version": "1.2.0", "columns_present": ["vx", "vy", "vz", "mass"]+
                [f"F{i}{j}" for i in range(3) for j in range(3)],
                "wall_clock_seconds": round(time.perf_counter()-self.started, 3),
                "hardware": torch.cuda.get_device_name(), "notes": " | ".join(notes)}
        (self.out/"meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run(scene_path: Path, out: Path):
    scene = json.loads(scene_path.read_text())
    sid = scene["scene_id"]
    unsupported = {"a1b_conservation_drifting_v1": "released fixed grid cannot contain 11.5 m drift",
        "a4_meltfront_v1": "no thermal state", "c2_blobhealth_melt_v1": "no thermal state",
        "a6_freefall_v1": "not the native HDF5 interchange scene", "c3_roundtrip_v1": "renderer not ported",
        "c5_triangle_shape_transport_v1": "not triangle based", "m1_material_validation_v1": "requires stress-strain export"}
    if sid in unsupported:
        raise SystemExit("NOT SUBMITTED: "+unsupported[sid])
    position_file = scene["geometry"].get("particle_positions_file")
    if not position_file:
        raise SystemExit("NOT SUBMITTED: no frozen positions")
    rest = np.load(scene_path.parent/position_file).astype(np.float64)
    extent = np.ptp(rest, axis=0)
    grid_lim = 1.2 if extent.max() > 0.9 else 1.0
    offset = np.full(3, grid_lim/2)-0.5*(rest.min(0)+rest.max(0))
    placed = rest+offset
    spacing = float(scene["geometry"]["particle_spacing_m"])
    grid = max(24, int(math.ceil(grid_lim/(2*spacing))))
    fps, duration = float(scene["simulation"]["output_fps"]), float(scene["simulation"]["duration_s"])
    material = scene["material"]
    if "regions" in material:
        interface = float(material["interface_x_m"])
        E = np.where(rest[:, 0] < interface, float(material["regions"][0]["youngs_modulus_pa"]),
                     float(material["regions"][1]["youngs_modulus_pa"]))
        nu = np.where(rest[:, 0] < interface, float(material["regions"][0]["poisson_ratio"]),
                      float(material["regions"][1]["poisson_ratio"]))
        density = float(material["regions"][0]["density_kg_m3"])
    else:
        E, nu = np.full(len(rest), float(material["youngs_modulus_pa"])), np.full(len(rest), float(material["poisson_ratio"]))
        density = float(material["density_kg_m3"])
    wave = math.sqrt(E.max()*(1-nu.min())/((1+nu.min())*(1-2*nu.min()))/density)
    frame_dt = 1/fps
    substeps = max(1, math.ceil(frame_dt/(0.25*(grid_lim/grid)/wave)))
    dt = frame_dt/substeps
    count, particle_volume = len(rest), volume(scene)/len(rest)
    tier_c = sid in {"c1_covariance_v1", "c2_blobhealth_shear_v1"}
    cov = None
    if tier_c:
        variance = (0.5*spacing)**2
        cov = torch.as_tensor(np.tile([variance, 0, 0, variance, 0, variance], (count, 1)), device="cuda", dtype=torch.float32)
    position = torch.as_tensor(placed, device="cuda", dtype=torch.float32)
    velocity = torch.as_tensor(initial_velocity(scene, rest), device="cuda", dtype=torch.float32)
    state = MPMStateStruct(); state.init(count, device=DEVICE, requires_grad=False)
    state.from_torch(position, torch.full((count,), particle_volume, device="cuda"), cov, velocity,
                     n_grid=grid, grid_lim=grid_lim, device=DEVICE, requires_grad=False)
    wp.launch(set_mat33_to_identity, dim=count, inputs=[state.particle_F], device=DEVICE)
    state.particle_C = from_torch_safe(torch.as_tensor(initial_C(scene, count), device="cuda"), dtype=wp.mat33, requires_grad=False)
    model = MPMModelStruct(); model.init(count, device=DEVICE, requires_grad=False)
    model.init_other_params(n_grid=grid, grid_lim=grid_lim, device=DEVICE)
    model.update_cov_with_F = bool(tier_c)
    solver = MPMWARPDiff(count, n_grid=grid, grid_lim=grid_lim, device=DEVICE)
    solver.set_parameters_dict(model, state, {"material": "jelly", "g": scene["simulation"]["gravity"],
                                              "density": density, "grid_v_damping_scale": 1.1}, device=DEVICE)
    density_t = torch.full((count,), density, device="cuda")
    state.reset_density(density_t, torch.ones(count, device="cuda", dtype=torch.int32), DEVICE, update_mass=True)
    solver.set_E_nu_from_torch(model, torch.as_tensor(E, device="cuda", dtype=torch.float32),
                               torch.as_tensor(nu, device="cuda", dtype=torch.float32), DEVICE)
    solver.prepare_mu_lam(model, state, DEVICE)
    prescribed = boundaries(scene, rest, offset)
    writer = Writer(out, offset, tier_c)
    mass = density*particle_volume
    writer.add(0, 0, state, mass)
    frames = round(duration*fps)
    for frame in range(1, frames+1):
        for _ in range(substeps):
            solver.p2g2p(model, state, 0, dt, device=DEVICE)
            for bc in prescribed:
                bc.apply(state, solver.time)
        writer.add(frame, frame/fps, state, mass)
        if frame % max(1, frames//10) == 0:
            print(f"{sid}: {frame}/{frames}", flush=True)
    writer.finish(["Released PhysDreamer physdreamer/warp_mpm solver.",
        f"Translation-only placement {offset.tolist()} inverted on export.",
        f"Native fixed grid [0,{grid_lim}]^3 at {grid}^3, dt={dt:.9g}, {substeps} substeps/frame.",
        "Tier C covariance is the released first-order native covariance update, not reconstructed by the adapter."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--scene", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); wp.init(); run(args.scene.resolve(), args.out.resolve())
