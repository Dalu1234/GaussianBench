"""C5 representation-native triangle shape-transport scorer."""

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
    if len(df) < 2 or not np.array_equal(df.frame.to_numpy(), np.arange(len(df))):
        raise common.SubmissionError("Need at least two contiguous frames from zero")
    if df.triangle_count.nunique() != 1:
        raise common.SubmissionError("Triangle identity/count changed across frames")

    c = scene["pass_criteria"]
    com = df[["com_x", "com_y", "com_z"]].to_numpy(float)
    travel = float(np.max(np.linalg.norm(com - com[0], axis=1)))
    measured = {
        "triangle_count": int(df.triangle_count.iloc[0]),
        "max_com_travel": travel,
        "max_edge_rel_error_median": float(df.edge_rel_error_median.max()),
        "max_edge_rel_error_p95": float(df.edge_rel_error_p95.max()),
        "max_area_rel_error_median": float(df.area_rel_error_median.max()),
        "max_area_rel_error_p95": float(df.area_rel_error_p95.max()),
        "max_nonfinite_fraction": float(df.nonfinite_fraction.max()),
    }
    if travel < c["min_com_travel"]:
        return common.verdict_from_scene(
            scene, common.STATUS_INVALID, measured=measured,
            tolerance={"min_com_travel": c["min_com_travel"]},
            notes=["The object did not undergo enough motion to test transport."])
    ok = (
        measured["max_edge_rel_error_median"] <= c["edge_median_max"] and
        measured["max_edge_rel_error_p95"] <= c["edge_p95_max"] and
        measured["max_area_rel_error_median"] <= c["area_median_max"] and
        measured["max_area_rel_error_p95"] <= c["area_p95_max"] and
        measured["max_nonfinite_fraction"] == 0.0
    )
    return common.verdict_from_scene(
        scene, common.STATUS_PASS if ok else common.STATUS_FAIL,
        measured=measured,
        reference={"edge_and_area_relative_error": 0.0},
        tolerance={k: c[k] for k in (
            "edge_median_max", "edge_p95_max", "area_median_max", "area_p95_max")})
