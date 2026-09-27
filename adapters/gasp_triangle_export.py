"""Convert GASP triangle-soup checkpoints into compact GaussianBench metrics.

The stable triangle ordering in GASP makes edge and area transport directly
auditable without copying multi-gigabyte ``.pt`` trajectories into a result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def triangle_quantities(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.stack(
        [np.linalg.norm(x[:, 1] - x[:, 0], axis=1),
         np.linalg.norm(x[:, 2] - x[:, 1], axis=1),
         np.linalg.norm(x[:, 0] - x[:, 2], axis=1)], axis=1)
    areas = 0.5 * np.linalg.norm(
        np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), axis=1)
    return edges, areas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--triangles-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--system-name", default="GASP")
    ap.add_argument("--intervention", default="none")
    args = ap.parse_args()

    files = sorted(args.triangles_dir.glob("*.pt"))
    if len(files) < 2:
        raise SystemExit("Need at least two GASP triangle checkpoints")

    rest = torch.load(files[0], map_location="cpu", weights_only=False)
    rest = np.asarray(rest, dtype=np.float64).reshape(-1, 3, 3)
    edge0, area0 = triangle_quantities(rest)
    valid_edge = edge0 > 1e-12
    valid_area = area0 > 1e-16
    rows = []
    for frame, path in enumerate(files):
        x = torch.load(path, map_location="cpu", weights_only=False)
        x = np.asarray(x, dtype=np.float64).reshape(-1, 3, 3)
        if x.shape != rest.shape:
            raise SystemExit(f"Triangle count changed in {path}")
        edge, area = triangle_quantities(x)
        er = np.abs(edge[valid_edge] / edge0[valid_edge] - 1.0)
        ar = np.abs(area[valid_area] / area0[valid_area] - 1.0)
        flat = x.reshape(-1, 3)
        rows.append({
            "frame": frame,
            "source_frame": int(path.stem),
            "com_x": float(flat[:, 0].mean()),
            "com_y": float(flat[:, 1].mean()),
            "com_z": float(flat[:, 2].mean()),
            "bbox_volume": float(np.prod(np.ptp(flat, axis=0))),
            "edge_rel_error_median": float(np.median(er)),
            "edge_rel_error_p95": float(np.quantile(er, 0.95)),
            "area_rel_error_median": float(np.median(ar)),
            "area_rel_error_p95": float(np.quantile(ar, 0.95)),
            "nonfinite_fraction": float(1.0 - np.isfinite(flat).mean()),
            "triangle_count": int(x.shape[0]),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "triangle_metrics.csv", index=False)
    (args.output_dir / "meta.json").write_text(json.dumps({
        "system": args.system_name,
        "representation": "GASP triangle soup",
        "source_directory": str(args.triangles_dir.resolve()),
        "intervention": args.intervention,
        "stable_triangle_order": True,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
