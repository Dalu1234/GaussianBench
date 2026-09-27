"""C7 representation-native supported-rest stability scorer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import common


REQUIRED = {
    "frame", "com_x", "com_y", "com_z", "bbox_volume",
    "edge_rel_error_median", "edge_rel_error_p95", "area_rel_error_median",
    "area_rel_error_p95", "nonfinite_fraction", "triangle_count",
}


def score(scene: dict, results_dir: Path, scenes_dir: Path) -> common.Verdict:
    del scenes_dir
    path = results_dir / "triangle_metrics.csv"
    if not path.exists():
        raise common.SubmissionError(f"Missing {path.name} in {results_dir}")
    df = pd.read_csv(path)
    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        raise common.SubmissionError(f"triangle_metrics.csv missing {missing}")
    if not np.array_equal(df.frame.to_numpy(), np.arange(len(df))):
        raise common.SubmissionError("Frames must be contiguous from zero")
    if df.triangle_count.nunique() != 1:
        raise common.SubmissionError("Triangle identity/count changed across frames")

    c = scene["pass_criteria"]
    if len(df) < c["min_frames"]:
        raise common.SubmissionError(
            f"Need at least {c['min_frames']} frames, got {len(df)}")

    length_scale = float(np.cbrt(df.bbox_volume.iloc[0]))
    if not np.isfinite(length_scale) or length_scale <= 0:
        raise common.SubmissionError("bbox_volume at frame 0 must be positive")

    settle = int(np.ceil(c["settle_frac"] * len(df)))
    settle = min(settle, len(df) - 2)
    com = df[["com_x", "com_y", "com_z"]].to_numpy(float)
    tail = com[settle:]
    drift = float(np.linalg.norm(tail - tail[0], axis=1).max() / length_scale)

    tail_df = df.iloc[settle:]
    measured = {
        "triangle_count": int(df.triangle_count.iloc[0]),
        "settle_frame": settle,
        "body_length_scale": length_scale,
        "tail_com_drift_rel": drift,
        "tail_max_edge_rel_error_median": float(tail_df.edge_rel_error_median.max()),
        "tail_max_edge_rel_error_p95": float(tail_df.edge_rel_error_p95.max()),
        "tail_max_area_rel_error_median": float(tail_df.area_rel_error_median.max()),
        "tail_max_area_rel_error_p95": float(tail_df.area_rel_error_p95.max()),
        "max_nonfinite_fraction": float(df.nonfinite_fraction.max()),
    }
    ok = (
        measured["tail_com_drift_rel"] <= c["tail_com_drift_max_rel"] and
        measured["tail_max_edge_rel_error_median"] <= c["edge_median_max"] and
        measured["tail_max_edge_rel_error_p95"] <= c["edge_p95_max"] and
        measured["tail_max_area_rel_error_median"] <= c["area_median_max"] and
        measured["tail_max_area_rel_error_p95"] <= c["area_p95_max"] and
        measured["max_nonfinite_fraction"] == 0.0
    )
    return common.verdict_from_scene(
        scene, common.STATUS_PASS if ok else common.STATUS_FAIL,
        measured=measured,
        reference={"tail_com_drift": 0.0, "edge_and_area_relative_error": 0.0},
        tolerance={k: c[k] for k in (
            "tail_com_drift_max_rel", "edge_median_max", "edge_p95_max",
            "area_median_max", "area_p95_max")})
