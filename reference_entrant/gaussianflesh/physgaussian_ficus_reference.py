"""Run and compare a state-level reproduction of PhysGaussian's ficus config."""

import argparse
import json
import logging
import os
import sys
import types
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parent
DEFAULT_PHYS_ROOT = Path(os.environ.get("PHYSGAUSSIAN_ROOT", REPO.parent / "PhysGaussian"))
LOGGER = logging.getLogger(__name__)

DEFAULT_PLY = (
    DEFAULT_PHYS_ROOT
    / "model"
    / "ficus_whitebg-trained"
    / "point_cloud"
    / "iteration_30000"
    / "point_cloud.ply"
)


def load_ficus_positions(path):
    with open(path, "rb") as stream:
        properties = []
        vertex_count = None
        while True:
            line = stream.readline().decode("ascii").strip()
            if line.startswith("format") and "binary_little_endian" not in line:
                raise ValueError("Only binary little-endian PLY files are supported")
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property"):
                properties.append(line.split()[-1])
            elif line == "end_header":
                break
        if vertex_count is None:
            raise ValueError("PLY has no vertex element")
        values = np.frombuffer(
            stream.read(vertex_count * len(properties) * 4), dtype="<f4"
        ).reshape(vertex_count, len(properties))

    index = {name: i for i, name in enumerate(properties)}
    opacity = 1.0 / (1.0 + np.exp(-values[:, index["opacity"]]))
    pos = values[opacity > 0.02][:, [index["x"], index["y"], index["z"]]].copy()
    lo = pos.min(axis=0)
    hi = pos.max(axis=0)
    pos = (pos - 0.5 * (lo + hi)) / np.max(hi - lo) + 1.0
    return pos.astype(np.float32)


def particle_volumes(pos, dx=0.04):
    cells = np.floor(pos / dx).astype(np.int32)
    _, inverse, counts = np.unique(
        cells, axis=0, return_inverse=True, return_counts=True
    )
    return ((dx ** 3) / counts[inverse]).astype(np.float32)


def save_phys_state(solver, out_dir, frame, wp):
    state = solver.mpm_state
    np.savez_compressed(
        out_dir / f"state_{frame:04d}.npz",
        x=wp.to_torch(state.particle_x).detach().cpu().numpy(),
        v=wp.to_torch(state.particle_v).detach().cpu().numpy(),
        F=wp.to_torch(state.particle_F_trial).detach().cpu().numpy(),
        C=wp.to_torch(state.particle_C).detach().cpu().numpy(),
    )
    LOGGER.info("[phys-reference] saved frame %s", frame)


def run_phys(args):
    phys_root = Path(args.phys_root).resolve()
    sys.path.insert(0, str(phys_root))
    sys.path.insert(0, str(phys_root / "mpm_solver_warp"))

    import torch
    import warp as wp

    if not hasattr(wp.types, "float32"):
        wp.types.float32 = wp.float32
    if "warp.torch" not in sys.modules:
        sys.modules["warp.torch"] = types.ModuleType("warp.torch")

    import mpm_solver_warp as solver_module

    wp.config.kernel_cache_dir = str(
        Path(__file__).resolve().parent / "comparison_runs" / ".warp_cache"
    )
    wp.config.use_precompiled_headers = False
    wp.init()
    # This checkout predates Warp 1.9's removal of the low-level ``owner``
    # array argument. Keep the compatibility adjustment outside PhysGaussian.
    solver_module.torch2warp_float = (
        lambda tensor, copy=False, dtype=None, dvc="cuda:0":
        wp.from_torch(tensor.contiguous(), dtype=wp.float32, requires_grad=False)
    )
    solver_module.torch2warp_vec3 = (
        lambda tensor, copy=False, dtype=None, dvc="cuda:0":
        wp.from_torch(tensor.contiguous(), dtype=wp.vec3, requires_grad=False)
    )
    solver_module.torch2warp_mat33 = (
        lambda tensor, copy=False, dtype=None, dvc="cuda:0":
        wp.from_torch(tensor.contiguous(), dtype=wp.mat33, requires_grad=False)
    )
    MPM_Simulator_WARP = solver_module.MPM_Simulator_WARP
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    pos = load_ficus_positions(args.ply)
    volume = particle_volumes(pos)
    pos_t = torch.from_numpy(pos).to("cuda:0")
    volume_t = torch.from_numpy(volume).to("cuda:0")

    solver = MPM_Simulator_WARP(10)
    solver.load_initial_data_from_torch(
        pos_t, volume_t, n_grid=50, grid_lim=2.0, device="cuda:0"
    )
    solver.set_parameters_dict(
        {
            "material": "jelly",
            "E": 2.0e6,
            "nu": 0.4,
            "density": 200.0,
            "g": [0.0, 0.0, 0.0],
            "grid_v_damping_scale": 0.9999,
            "rpic_damping": 0.0,
            "additional_material_params": [
                {
                    "point": [1.0, 1.0, 1.5],
                    "size": [1.0, 1.0, 0.5],
                    "E": 2.0e6,
                    "nu": 0.4,
                    "density": 70.0,
                }
            ],
        },
        device="cuda:0",
    )
    solver.set_velocity_on_cuboid(
        point=[1.0, 1.0, 0.5],
        size=[0.5, 0.5, 0.28],
        velocity=[0.0, 0.0, 0.0],
        start_time=0.0,
        end_time=1000.0,
        reset=1,
    )
    solver.add_impulse_on_particles(
        force=[-0.18, 0.0, 0.0],
        dt=1.0e-4,
        num_dt=1,
        start_time=0.0,
        device="cuda:0",
    )
    solver.finalize_mu_lam(device="cuda:0")

    samples = {0, 1, 2, 5, 10, 25, 50, 75, 100, args.frames}
    samples = {frame for frame in samples if 0 <= frame <= args.frames}
    save_phys_state(solver, out_dir, 0, wp)
    for frame in range(1, args.frames + 1):
        for _ in range(400):
            solver.p2g2p(frame - 1, 1.0e-4, device="cuda:0")
        if frame in samples:
            save_phys_state(solver, out_dir, frame, wp)

    density = wp.to_torch(solver.mpm_state.particle_density).cpu().numpy()
    mass = wp.to_torch(solver.mpm_state.particle_mass).cpu().numpy()
    metadata = {
        "backend": "physgaussian_warp",
        "particles": int(len(pos)),
        "frames": int(args.frames),
        "frame_dt": 0.04,
        "substep_dt": 1.0e-4,
        "substeps_per_frame": 400,
        "density_70_count": int(np.count_nonzero(density == 70.0)),
        "density_200_count": int(np.count_nonzero(density == 200.0)),
        "total_volume": float(volume.sum()),
        "total_mass": float(mass.sum()),
        "sample_frames": sorted(samples),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def run_phys_video(args):
    phys_root = Path(args.phys_root).resolve()
    sys.path.insert(0, str(phys_root))
    sys.path.insert(0, str(phys_root / "mpm_solver_warp"))

    import torch
    import warp as wp

    if not hasattr(wp.types, "float32"):
        wp.types.float32 = wp.float32
    if "warp.torch" not in sys.modules:
        sys.modules["warp.torch"] = types.ModuleType("warp.torch")

    import mpm_solver_warp as solver_module
    from matched_ficus_video import MatchedFicusRenderer

    wp.config.kernel_cache_dir = str(
        Path(__file__).resolve().parent / "comparison_runs" / ".warp_cache"
    )
    wp.config.use_precompiled_headers = False
    wp.init()
    solver_module.torch2warp_float = (
        lambda tensor, copy=False, dtype=None, dvc="cuda:0":
        wp.from_torch(tensor.contiguous(), dtype=wp.float32, requires_grad=False)
    )
    solver_module.torch2warp_vec3 = (
        lambda tensor, copy=False, dtype=None, dvc="cuda:0":
        wp.from_torch(tensor.contiguous(), dtype=wp.vec3, requires_grad=False)
    )
    solver_module.torch2warp_mat33 = (
        lambda tensor, copy=False, dtype=None, dvc="cuda:0":
        wp.from_torch(tensor.contiguous(), dtype=wp.mat33, requires_grad=False)
    )

    pos = load_ficus_positions(args.ply)
    volume = particle_volumes(pos)
    solver = solver_module.MPM_Simulator_WARP(10)
    solver.load_initial_data_from_torch(
        torch.from_numpy(pos).to("cuda:0"),
        torch.from_numpy(volume).to("cuda:0"),
        n_grid=50,
        grid_lim=2.0,
        device="cuda:0",
    )
    solver.set_parameters_dict(
        {
            "material": "jelly",
            "E": 2.0e6,
            "nu": 0.4,
            "density": 200.0,
            "g": [0.0, 0.0, -5.0],
            "grid_v_damping_scale": 0.9999,
            "rpic_damping": 0.0,
        },
        device="cuda:0",
    )
    solver.add_bounding_box(start_time=0.0, end_time=1000.0)
    solver.finalize_mu_lam(device="cuda:0")

    renderer = MatchedFicusRenderer(args.ply, args.output, fps=30.0, size=800)
    renderer.write(pos, "PhysGaussian Warp MPM", 0)
    for frame in range(1, args.frames + 1):
        for _ in range(80):
            solver.p2g2p(frame - 1, 1.0e-4, device="cuda:0")
        current = wp.to_torch(solver.mpm_state.particle_x).detach().cpu().numpy()
        renderer.write(current, "PhysGaussian Warp MPM", frame)
        if frame % 10 == 0:
            LOGGER.info("[phys-video] rendered frame %s/%s", frame, args.frames)
    renderer.close()
    LOGGER.info("[phys-video] video: %s", args.output)


def summarize_error(delta):
    per_particle = np.linalg.norm(delta.reshape(len(delta), -1), axis=1)
    return {
        "rmse_components": float(np.sqrt(np.mean(delta.astype(np.float64) ** 2))),
        "mean_norm": float(per_particle.mean()),
        "p95_norm": float(np.percentile(per_particle, 95)),
        "max_norm": float(per_particle.max()),
    }


def compare(args):
    phys_dir = Path(args.phys)
    ours_dir = Path(args.ours)
    frames = sorted(
        int(path.stem.split("_")[-1])
        for path in phys_dir.glob("state_*.npz")
        if (ours_dir / path.name).exists()
    )
    if not frames:
        raise ValueError("No matching state checkpoints found")

    results = []
    for frame in frames:
        with np.load(phys_dir / f"state_{frame:04d}.npz") as phys:
            with np.load(ours_dir / f"state_{frame:04d}.npz") as ours:
                if phys["x"].shape != ours["x"].shape:
                    raise ValueError(
                        f"Frame {frame} shape mismatch: "
                        f"{phys['x'].shape} != {ours['x'].shape}"
                    )
                result = {
                    "frame": frame,
                    "time": frame * 0.04,
                    "position": summarize_error(ours["x"] - phys["x"]),
                    "velocity": summarize_error(ours["v"] - phys["v"]),
                    "F": summarize_error(ours["F"] - phys["F"]),
                    "center_of_mass_delta": (
                        ours["x"].mean(axis=0) - phys["x"].mean(axis=0)
                    ).tolist(),
                }
                results.append(result)
                LOGGER.info(
                    "[compare] frame=%3d x_mean=%.6e v_mean=%.6e F_mean=%.6e",
                    frame,
                    result["position"]["mean_norm"],
                    result["velocity"]["mean_norm"],
                    result["F"]["mean_norm"],
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("[compare] wrote %s", output)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    phys = sub.add_parser("phys")
    phys.add_argument("--phys-root", default=str(DEFAULT_PHYS_ROOT))
    phys.add_argument("--ply", default=str(DEFAULT_PLY))
    phys.add_argument("--output", required=True)
    phys.add_argument("--frames", type=int, default=10)
    phys.set_defaults(func=run_phys)

    video = sub.add_parser("video")
    video.add_argument("--phys-root", default=str(DEFAULT_PHYS_ROOT))
    video.add_argument("--ply", default=str(DEFAULT_PLY))
    video.add_argument("--output", required=True)
    video.add_argument("--frames", type=int, default=125)
    video.set_defaults(func=run_phys_video)

    comp = sub.add_parser("compare")
    comp.add_argument("--phys", required=True)
    comp.add_argument("--ours", required=True)
    comp.add_argument("--output", required=True)
    comp.set_defaults(func=compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
