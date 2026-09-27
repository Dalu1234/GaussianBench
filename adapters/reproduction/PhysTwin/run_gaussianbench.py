"""Run PhysTwin's released Warp spring-mass simulator on applicable scenes."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO))
from qqtt.utils import cfg  # noqa: E402
from qqtt.model.diff_simulator.spring_mass_warp import SpringMassSystemWarp  # noqa: E402


def native_body() -> torch.Tensor:
    """Deterministic native particle body; A6 intentionally ships no geometry."""
    axis = torch.linspace(-0.05, 0.05, 6)
    points = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    points[:, 2] += 1.0
    return points.cuda().float()


def native_springs(points: torch.Tensor, neighbors: int = 16):
    distance = torch.cdist(points, points)
    indices = torch.topk(distance, neighbors+1, largest=False).indices[:, 1:]
    edge_set = set()
    for i, row in enumerate(indices.detach().cpu().tolist()):
        for j in row:
            edge_set.add((min(i, j), max(i, j)))
    edges = torch.tensor(sorted(edge_set), device="cuda", dtype=torch.int32)
    lengths = torch.linalg.norm(points[edges[:, 0].long()]-points[edges[:, 1].long()], dim=1)
    return edges, lengths


def run(scene_path: Path, out: Path):
    scene = json.loads(scene_path.read_text())
    if scene["scene_id"] != "a6_freefall_v1":
        raise SystemExit("NOT SUBMITTED: released PhysTwin dynamics hard-code gravity and learn scene-specific springs")
    cfg.use_graph = False
    cfg.data_type = "synthetic"
    cfg.collision_learn = False
    cfg.device = "cuda:0"
    points = native_body()
    springs, rest_lengths = native_springs(points)
    count = len(points)
    masses = torch.full((count,), 1.0/count, device="cuda")
    fps = float(scene["simulation"]["output_fps"])
    frame_dt = 1/fps
    dt = 5e-5
    substeps = round(frame_dt/dt)
    dummy_gt = torch.stack([points, points]).detach().clone()
    system = SpringMassSystemWarp(
        init_vertices=points, init_springs=springs,
        init_rest_lengths=rest_lengths, init_masses=masses,
        dt=dt, num_substeps=substeps, spring_Y=3000.0,
        collide_elas=0.0, collide_fric=0.0,
        dashpot_damping=0.0, drag_damping=0.0,
        num_object_points=count, num_surface_points=count,
        num_original_points=count, controller_points=None,
        reverse_z=False, spring_Y_min=0.0, spring_Y_max=1e5,
        gt_object_points=dummy_gt, self_collision=False,
        disable_backward=True)
    system.set_init_state(system.wp_init_vertices, system.wp_init_velocities, pure_inference=True)
    rows = []
    started = time.perf_counter()

    def record(frame: int, wp_x, wp_v):
        x = wp.to_torch(wp_x).detach().cpu().numpy()
        v = wp.to_torch(wp_v).detach().cpu().numpy()
        extent = np.ptp(x, axis=0)
        finite = np.isfinite(x).all(1) & np.isfinite(v).all(1)
        rows.append({"frame": frame, "time": frame/fps,
            "com_x": float(x[:, 0].mean()), "com_y": float(x[:, 1].mean()), "com_z": float(x[:, 2].mean()),
            "com_vx": float(v[:, 0].mean()), "com_vy": float(v[:, 1].mean()), "com_vz": float(v[:, 2].mean()),
            "bbox_volume": float(np.prod(extent)),
            "detF_nonpositive_fraction": 0.0,
            "nonfinite_fraction": float(1-finite.mean()), "particle_count": count})

    record(0, system.wp_states[0].wp_x, system.wp_states[0].wp_v)
    frames = round(float(scene["simulation"]["duration_s"])*fps)
    for frame in range(1, frames+1):
        system.step()
        final = system.wp_states[-1]
        record(frame, final.wp_x, final.wp_v)
        if frame < frames:
            system.set_init_state(final.wp_x, final.wp_v, pure_inference=True)
    out.mkdir(parents=True, exist_ok=True)
    with (out/"body_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    meta = {"system_name": "PhysTwin", "system_version": commit, "spec_version": "1.2.0",
        "wall_clock_seconds": round(time.perf_counter()-started, 3),
        "hardware": torch.cuda.get_device_name(),
        "notes": {"text": ("Released SpringMassSystemWarp on a deterministic native 216-particle body, "
            f"{len(springs)} released-radius-style undirected springs, dt={dt}, {substeps} substeps/frame. "
            "PhysTwin has no deformation-gradient state; detF_nonpositive_fraction is the empty-set fraction 0, "
            "while bounding-volume and nonfinite health remain directly measured.")}}
    (out/"meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--scene", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); run(args.scene.resolve(), args.out.resolve())
