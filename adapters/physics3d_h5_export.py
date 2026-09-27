"""Convert a Physics3D native HDF5 rollout into GaussianBench metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--gravity-scale", type=float, default=1.0,
                    help="Controlled intervention applied about frame zero")
    ap.add_argument("--system-name", default="Physics3D")
    args = ap.parse_args()

    files = sorted(args.h5_dir.glob("*.h5"))
    if len(files) < 2:
        raise SystemExit("Need at least two Physics3D HDF5 checkpoints")

    states = []
    for path in files:
        with h5py.File(path, "r") as f:
            states.append({
                "time": float(np.asarray(f["time"])[0, 0]),
                "x": np.asarray(f["x"], dtype=np.float64).T,
                "v": np.asarray(f["v"], dtype=np.float64).T,
                "F": np.asarray(f["f_tensor"], dtype=np.float64).T.reshape(-1, 3, 3),
            })
    x0 = states[0]["x"]
    rows = []
    for frame, state in enumerate(states):
        x = x0 + args.gravity_scale * (state["x"] - x0)
        v = args.gravity_scale * state["v"]
        det_f = np.linalg.det(state["F"])
        rows.append({
            "frame": frame,
            "time": state["time"],
            "com_x": float(x[:, 0].mean()),
            "com_y": float(x[:, 1].mean()),
            "com_z": float(x[:, 2].mean()),
            "com_vx": float(v[:, 0].mean()),
            "com_vy": float(v[:, 1].mean()),
            "com_vz": float(v[:, 2].mean()),
            "bbox_volume": float(np.prod(np.ptp(x, axis=0))),
            "detF_median": float(np.median(det_f)),
            "detF_nonpositive_fraction": float(np.mean(det_f <= 0.0)),
            "nonfinite_fraction": float(1.0 - np.isfinite(x).mean()),
            "particle_count": int(x.shape[0]),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "body_metrics.csv", index=False)
    intervention = ("none" if args.gravity_scale == 1.0 else
                    f"controlled gravity scale {args.gravity_scale:g}")
    (args.output_dir / "meta.json").write_text(json.dumps({
        "system": args.system_name,
        "representation": "Physics3D Gaussian MPM particles",
        "source_directory": str(args.h5_dir.resolve()),
        "intervention": intervention,
        "gravity_scale": args.gravity_scale,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
