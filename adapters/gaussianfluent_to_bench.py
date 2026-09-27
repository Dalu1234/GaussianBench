"""Convert GaussianFluent native HDF5 frame dumps to GaussianBench files.

GaussianFluent's ``save_data_at_frame`` writes x as (3,N), f_tensor as (9,N),
and v as (3,N). This adapter does not invent covariance state: callers must pass
native covariance exports when grading C1/C2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def _frames(native_dir: Path):
    files = sorted(native_dir.glob("sim_*.h5"))
    if not files:
        raise FileNotFoundError(f"no sim_*.h5 files under {native_dir}")
    for frame, path in enumerate(files):
        with h5py.File(path, "r") as h:
            x = np.asarray(h["x"], dtype=np.float32).T
            v = np.asarray(h["v"], dtype=np.float32).T if "v" in h else np.zeros_like(x)
            F = np.asarray(h["f_tensor"], dtype=np.float32).T.reshape(-1, 3, 3)
            t = float(np.asarray(h["time"])[0, 0]) if "time" in h else float(frame)
        yield frame, t, x, v, F


def convert(native_dir: Path, output_dir: Path, rest_cov: Path | None = None):
    rows = []
    cov_frames = []
    n_particles = None
    for frame, time, x, v, F in _frames(native_dir):
        n_particles = x.shape[0] if n_particles is None else n_particles
        if x.shape[0] != n_particles:
            raise ValueError("GaussianFluent frame particle count changed; stable IDs required")
        for pid in range(n_particles):
            rows.append({"frame": frame, "time": time, "particle_id": pid,
                         "x": float(x[pid, 0]), "y": float(x[pid, 1]), "z": float(x[pid, 2]),
                         "vx": float(v[pid, 0]), "vy": float(v[pid, 1]), "vz": float(v[pid, 2]),
                         "F00": float(F[pid, 0, 0]), "F01": float(F[pid, 0, 1]), "F02": float(F[pid, 0, 2]),
                         "F10": float(F[pid, 1, 0]), "F11": float(F[pid, 1, 1]), "F12": float(F[pid, 1, 2]),
                         "F20": float(F[pid, 2, 0]), "F21": float(F[pid, 2, 1]), "F22": float(F[pid, 2, 2])})
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_dir / "trajectory.parquet", index=False)
    meta = {"system": "GaussianFluent", "adapter": "gaussianfluent_to_bench",
            "native_export": "Warp MPM HDF5 sim_*.h5", "covariance_export": bool(rest_cov),
            "notes": {"domain_handling": "other", "deformation_gradient": "native particle_F"}}
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if rest_cov is not None:
        cov = np.load(rest_cov)
        if cov.shape != (n_particles, 6):
            raise ValueError(f"rest covariance must have shape ({n_particles}, 6), got {cov.shape}")
        np.save(output_dir / "rest_covariance.npy", cov)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--rest-cov", type=Path)
    args = ap.parse_args()
    print(f"wrote {convert(args.native_dir, args.output_dir, args.rest_cov)} trajectory rows")


if __name__ == "__main__":
    main()
