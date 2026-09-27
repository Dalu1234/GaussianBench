"""C8 representation-native impact robustness and recovery scorer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import common


REQUIRED = {
    "frame", "com_x", "com_y", "com_z", "edge_rel_error_median",
    "edge_rel_error_p95", "area_rel_error_median", "area_rel_error_p95",
    "nonfinite_fraction", "triangle_count",
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

    com = df[["com_x", "com_y", "com_z"]].to_numpy(float)
    travel = float(np.max(np.linalg.norm(com - com[0], axis=1)))
    peak_area_p95 = float(df.area_rel_error_p95.max())
    tail_start = int(np.floor((1.0 - c["tail_frac"]) * len(df)))
    tail_start = min(tail_start, len(df) - 1)
    tail = df.iloc[tail_start:]

    measured = {
        "triangle_count": int(df.triangle_count.iloc[0]),
        "max_com_travel": travel,
        "peak_area_rel_error_p95": peak_area_p95,
        "max_edge_rel_error_p95": float(df.edge_rel_error_p95.max()),
        "tail_best_edge_rel_error_median": float(tail.edge_rel_error_median.min()),
        "tail_best_area_rel_error_median": float(tail.area_rel_error_median.min()),
        "max_nonfinite_fraction": float(df.nonfinite_fraction.max()),
    }
    gates_invalid = []
    if travel < c["min_com_travel"]:
        gates_invalid.append(
            "The body did not travel far enough to have hit the floor.")
    if peak_area_p95 < c["min_peak_area_p95"]:
        gates_invalid.append(
            "The impact did not deform the body enough to test robustness; "
            "a too-gentle run cannot claim this capability.")
    if gates_invalid:
        return common.verdict_from_scene(
            scene, common.STATUS_INVALID, measured=measured,
            tolerance={"min_com_travel": c["min_com_travel"],
                       "min_peak_area_p95": c["min_peak_area_p95"]},
            notes=gates_invalid)
    ok = (
        measured["max_edge_rel_error_p95"] <= c["transient_edge_p95_max"] and
        measured["tail_best_edge_rel_error_median"] <= c["tail_edge_median_max"] and
        measured["tail_best_area_rel_error_median"] <= c["tail_area_median_max"] and
        measured["max_nonfinite_fraction"] == 0.0
    )
    return common.verdict_from_scene(
        scene, common.STATUS_PASS if ok else common.STATUS_FAIL,
        measured=measured,
        reference={"tail_shape_error": "returns below the recovery bound",
                   "transient": "bounded, finite, identity-stable"},
        tolerance={k: c[k] for k in (
            "transient_edge_p95_max", "tail_edge_median_max",
            "tail_area_median_max")})
