"""A6 free-fall solution verification and deformation-health scorer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import common


REQUIRED = {
    "frame", "time", "com_x", "com_y", "com_z", "com_vx", "com_vy",
    "com_vz", "bbox_volume", "detF_nonpositive_fraction",
    "nonfinite_fraction", "particle_count",
}


def score(scene: dict, results_dir: Path, scenes_dir: Path) -> common.Verdict:
    del scenes_dir
    path = results_dir / "body_metrics.csv"
    if not path.exists():
        raise common.SubmissionError(f"Missing {path.name} in {results_dir}")
    df = pd.read_csv(path)
    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        raise common.SubmissionError(f"body_metrics.csv missing {missing}")
    if len(df) < 2 or not np.array_equal(df.frame.to_numpy(), np.arange(len(df))):
        raise common.SubmissionError("Need at least two contiguous frames from zero")
    t = df.time.to_numpy(float)
    if t[0] != 0.0 or np.any(np.diff(t) <= 0):
        raise common.SubmissionError("Times must start at zero and increase")

    g = np.asarray(scene["simulation"]["gravity"], dtype=float)
    axis = int(np.argmax(np.abs(g)))
    x = df[["com_x", "com_y", "com_z"]].to_numpy(float)[:, axis]
    v = df[["com_vx", "com_vy", "com_vz"]].to_numpy(float)[:, axis]
    expected_dx = 0.5 * g[axis] * t[1:] ** 2
    expected_v = g[axis] * t[1:]
    pos_error = np.abs((x[1:] - x[0]) - expected_dx) / np.abs(expected_dx)
    vel_error = np.abs(v[1:] - expected_v) / np.abs(expected_v)
    bbox = df.bbox_volume.to_numpy(float)
    bbox_drift = np.abs(bbox / bbox[0] - 1.0)
    measured = {
        "particle_count": int(df.particle_count.iloc[0]),
        "max_com_position_rel_error": float(pos_error.max()),
        "max_com_velocity_rel_error": float(vel_error.max()),
        "max_bbox_volume_rel_drift": float(bbox_drift.max()),
        "max_detF_nonpositive_fraction_after_initial":
            float(df.detF_nonpositive_fraction.iloc[1:].max()),
        "max_nonfinite_fraction": float(df.nonfinite_fraction.max()),
    }
    c = scene["pass_criteria"]
    ok = (
        measured["max_com_position_rel_error"] <= c["com_position_rel_error_max"] and
        measured["max_com_velocity_rel_error"] <= c["com_velocity_rel_error_max"] and
        measured["max_bbox_volume_rel_drift"] <= c["bbox_volume_rel_drift_max"] and
        measured["max_detF_nonpositive_fraction_after_initial"] <=
            c["detF_nonpositive_fraction_max"] and
        measured["max_nonfinite_fraction"] == 0.0
    )
    return common.verdict_from_scene(
        scene, common.STATUS_PASS if ok else common.STATUS_FAIL,
        measured=measured,
        reference={"com_position": "x0 + 0.5*g*t^2", "com_velocity": "g*t"},
        tolerance={k: c[k] for k in (
            "com_position_rel_error_max", "com_velocity_rel_error_max",
            "bbox_volume_rel_drift_max", "detF_nonpositive_fraction_max")})
