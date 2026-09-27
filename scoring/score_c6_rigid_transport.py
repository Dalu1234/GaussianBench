"""C6 representation-native Galilean rigid-transport scorer."""

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


def _load(results_dir: Path) -> pd.DataFrame:
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
    return df


def score(scene: dict, results_dir: Path, scenes_dir: Path) -> common.Verdict:
    del scenes_dir
    df = _load(results_dir)
    c = scene["pass_criteria"]
    if len(df) < c["min_frames"]:
        raise common.SubmissionError(
            f"Need at least {c['min_frames']} frames, got {len(df)}")

    com = df[["com_x", "com_y", "com_z"]].to_numpy(float)
    disp = com - com[0]
    travel = float(np.linalg.norm(disp[-1]))

    direction = np.asarray(
        scene["initial_conditions"]["uniform_velocity_direction"], float)
    direction = direction / np.linalg.norm(direction)

    # Straightness: max perpendicular deviation from the chord, over travel.
    chord = disp[-1] / max(travel, 1e-300)
    perp = disp - np.outer(disp @ chord, chord)
    straightness = float(np.linalg.norm(perp, axis=1).max() / max(travel, 1e-300))

    # Speed constancy over per-frame COM displacements.
    steps = np.linalg.norm(np.diff(com, axis=0), axis=1)
    med = float(np.median(steps))
    speed_variation = float(np.abs(steps - med).max() / max(med, 1e-300))

    cosine = float(chord @ direction)

    measured = {
        "triangle_count": int(df.triangle_count.iloc[0]),
        "com_travel": travel,
        "direction_cosine": cosine,
        "path_straightness": straightness,
        "speed_variation": speed_variation,
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
            notes=["The body did not travel far enough to test transport."])
    ok = (
        measured["direction_cosine"] >= c["direction_cosine_min"] and
        measured["path_straightness"] <= c["path_straightness_max"] and
        measured["speed_variation"] <= c["speed_variation_max"] and
        measured["max_edge_rel_error_median"] <= c["edge_median_max"] and
        measured["max_edge_rel_error_p95"] <= c["edge_p95_max"] and
        measured["max_area_rel_error_median"] <= c["area_median_max"] and
        measured["max_area_rel_error_p95"] <= c["area_p95_max"] and
        measured["max_nonfinite_fraction"] == 0.0
    )
    return common.verdict_from_scene(
        scene, common.STATUS_PASS if ok else common.STATUS_FAIL,
        measured=measured,
        reference={"com_path": "straight line at constant speed along the "
                               "declared direction",
                   "edge_and_area_relative_error": 0.0},
        tolerance={k: c[k] for k in (
            "direction_cosine_min", "path_straightness_max",
            "speed_variation_max", "edge_median_max", "edge_p95_max",
            "area_median_max", "area_p95_max")})
