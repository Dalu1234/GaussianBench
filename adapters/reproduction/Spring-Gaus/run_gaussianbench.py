"""Run conservation scenes through Spring-Gaus's released spring-mass core.

Spring-Gaus learns scene-specific spring and mass fields and has no released
mapping from continuum E/nu to those fields. Consequently this adapter enters
only the A1 conservation scenes, whose references do not depend on a particular
constitutive calibration. It does not relabel uncalibrated runs as A2/A3/A5.
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
from yacs.config import CfgNode as CN

REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO))
from lib.models.spring_mass.Spring_Mass import Spring_Mass  # noqa: E402


def scene_volume(scene: dict) -> float:
    geometry = scene["geometry"]
    return 4*math.pi*float(geometry["radius_m"])**3/3


def make_config(fps: float, mass: float, stiffness: float) -> CN:
    cfg = CN(new_allowed=True)
    cfg.K_NEIGHBORS = 16
    cfg.K_BINDING = 8
    cfg.N_STEP = 1
    cfg.PRETRAINED = None
    cfg.FITTING = False
    cfg.STRETCH_RATIOS = 0.0
    cfg.RATIO_FACTOR = 1.0
    cfg.DAMPING = False
    cfg.FIX_MASS = True
    cfg.FIX_DAMP = True
    cfg.OPTIM_FIXED_DAMP = False
    cfg.FIX_K = True
    cfg.UNLINEAR_FORCE = False
    cfg.POWER = 0.0
    cfg.OPTIM_G = False
    cfg.SOFT_K = False
    cfg.SPRING_BC = False
    cfg.SINGLE_K = True
    cfg.OPTIM_DY_VELOCITY = False
    cfg.INIT_VELOCITY = [0.0, 0.0, 0.0]
    cfg.G = [0.0, 0.0, 0.0]
    cfg.DATA = CN(new_allowed=True)
    cfg.DATA.FREQ = fps
    cfg.DATA.DT = 1/fps
    cfg.DATA.BC = [[[0.0, -1.0e6, 0.0], [0.0, 1.0, 0.0]]]
    cfg.DATA.GLOBAL_K = stiffness
    cfg.DATA.GLOBAL_M = mass
    cfg.DATA.GLOBAL_DAMP = 0.0
    return cfg


def initial_velocity(scene: dict, rest: np.ndarray) -> np.ndarray:
    ic = scene["initial_conditions"]
    velocity = np.broadcast_to(np.asarray(ic["linear_velocity_m_s"], float), rest.shape).copy()
    omega = np.asarray(ic["angular_velocity_rad_s"], float)
    velocity += np.cross(omega[None], rest-rest.mean(0))
    return velocity


def run(scene_path: Path, out: Path):
    scene = json.loads(scene_path.read_text())
    sid = scene["scene_id"]
    if sid not in {"a1a_conservation_inplace_v1", "a1b_conservation_drifting_v1"}:
        raise SystemExit("NOT SUBMITTED: Spring-Gaus has no released continuum-material mapping for this frozen scene")
    rest = np.load(scene_path.parent/scene["geometry"]["particle_positions_file"]).astype(np.float32)
    count = len(rest)
    density = float(scene["material"]["density_kg_m3"])
    total_volume = scene_volume(scene)
    particle_mass = density*total_volume/count
    # Cauchy-scale initialization: E*V divided over directed spring lengths.
    # This is fixed before the run; A1's conservation reference is independent
    # of its value, so it is not fitted to the measured verdict.
    E = float(scene["material"]["youngs_modulus_pa"])
    stiffness = E*total_volume/(count*16)
    fps = float(scene["simulation"]["output_fps"])
    duration = float(scene["simulation"]["duration_s"])
    xyz = torch.as_tensor(rest, device="cuda")
    velocity = torch.as_tensor(initial_velocity(scene, rest), device="cuda", dtype=torch.float32)
    model = Spring_Mass(make_config(fps, particle_mass, stiffness), xyz,
                        init_velocity=[0.0, 0.0, 0.0]).cuda().eval()
    K = (10**model.global_k.reshape(1, 1).repeat(count, model.k_neighbors)) / (model.origin_len+model.eps)
    mass = 10**model.global_m
    rebound = torch.tensor(0.0, device="cuda")
    friction = torch.tensor(1.0, device="cuda")
    # Conservative stability bound for the released semi-implicit step.
    omega_bound = torch.sqrt((K.sum(1)/mass).max()).item()
    frame_dt = 1/fps
    substeps = max(1, math.ceil(frame_dt/(0.15/omega_bound)))
    dt = frame_dt/substeps
    rows = []
    started = time.perf_counter()

    def record(frame: int):
        x = xyz.detach().cpu().numpy(); v = velocity.detach().cpu().numpy()
        rows.append(pd.DataFrame({"frame": frame, "time": frame/fps,
            "particle_id": np.arange(count), "x": x[:, 0], "y": x[:, 1], "z": x[:, 2],
            "vx": v[:, 0], "vy": v[:, 1], "vz": v[:, 2], "mass": particle_mass}))

    record(0)
    frames = round(duration*fps)
    with torch.no_grad():
        for frame in range(1, frames+1):
            for _ in range(substeps):
                xyz, velocity = model.step(xyz, velocity, K, mass, rebound, friction, None, dt)
            record(frame)
            if frame % max(1, frames//10) == 0:
                print(f"{sid}: {frame}/{frames}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_parquet(out/"trajectory.parquet", index=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    meta = {"system_name": "Spring-Gaus", "system_version": commit, "spec_version": "1.2.0",
        "columns_present": ["vx", "vy", "vz", "mass"],
        "wall_clock_seconds": round(time.perf_counter()-started, 3),
        "hardware": torch.cuda.get_device_name(),
        "notes": {"domain_handling": "mesh_free", "text": (
                  "Released Spring_Mass.step with released k-nearest spring construction. "
                  f"16 neighbors, fixed pre-run stiffness {stiffness:.9g}, dt={dt:.9g}, {substeps} substeps/frame. "
                  "No checkpoint or verdict-fitted parameter. Only constitutive-calibration-independent A1 scenes entered.")}}
    (out/"meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--scene", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); run(args.scene.resolve(), args.out.resolve())
