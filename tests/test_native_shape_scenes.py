"""Synthetic PASS/FAIL/INVALID regression tests for C6, C7, and C8.

Each test constructs a triangle_metrics.csv with a known outcome and asserts
the scorer reproduces it, including the scene's natural controlled fault.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scoring import (score_c6_rigid_transport, score_c7_supported_settle,
                     score_c8_impact_recovery)


ROOT = Path(__file__).parents[1]


def load_scene(name):
    return json.loads((ROOT / "scenes" / name).read_text())


def write_metrics(out, com, edge_med=0.0, edge_p95=0.0, area_med=0.0,
                  area_p95=0.0, nonfinite=0.0, count=100, bbox=None):
    n = len(com)
    def col(v):
        return np.full(n, v, float) if np.isscalar(v) else np.asarray(v, float)
    data = {
        "frame": np.arange(n),
        "com_x": com[:, 0], "com_y": com[:, 1], "com_z": com[:, 2],
        "edge_rel_error_median": col(edge_med),
        "edge_rel_error_p95": col(edge_p95),
        "area_rel_error_median": col(area_med),
        "area_rel_error_p95": col(area_p95),
        "nonfinite_fraction": col(nonfinite),
        "triangle_count": np.full(n, count, int),
    }
    if bbox is not None:
        data["bbox_volume"] = col(bbox)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(out / "triangle_metrics.csv", index=False)


def test_c6_accepts_uniform_motion(tmp_path):
    scene = load_scene("c6_rigid_transport_v1.json")
    d = np.array([1.0, 0.6, 0.3]); d /= np.linalg.norm(d)
    t = np.linspace(0.0, 1.0, 40)
    com = np.outer(t, d) * 0.6
    write_metrics(tmp_path, com, edge_med=1e-4, edge_p95=1e-3)
    v = score_c6_rigid_transport.score(scene, tmp_path, ROOT / "scenes")
    assert v.status == "PASS"


def test_c6_rejects_gravity_left_on(tmp_path):
    # The natural fault: engine gravity not zeroed -> parabolic COM path.
    scene = load_scene("c6_rigid_transport_v1.json")
    d = np.array([1.0, 0.6, 0.3]); d /= np.linalg.norm(d)
    t = np.linspace(0.0, 1.0, 40)
    com = np.outer(t, d) * 0.6
    com[:, 2] -= 0.5 * 0.5 * t ** 2  # gravity sag
    write_metrics(tmp_path, com)
    v = score_c6_rigid_transport.score(scene, tmp_path, ROOT / "scenes")
    assert v.status == "FAIL"
    assert v.measured["path_straightness"] > scene["pass_criteria"]["path_straightness_max"]


def test_c6_rejects_shape_distortion(tmp_path):
    scene = load_scene("c6_rigid_transport_v1.json")
    d = np.array([1.0, 0.6, 0.3]); d /= np.linalg.norm(d)
    com = np.outer(np.linspace(0, 1, 40), d) * 0.6
    write_metrics(tmp_path, com, edge_med=0.2, edge_p95=0.4)
    assert score_c6_rigid_transport.score(
        scene, tmp_path, ROOT / "scenes").status == "FAIL"


def test_c6_short_travel_is_invalid(tmp_path):
    scene = load_scene("c6_rigid_transport_v1.json")
    d = np.array([1.0, 0.6, 0.3]); d /= np.linalg.norm(d)
    com = np.outer(np.linspace(0, 1, 40), d) * 0.01
    write_metrics(tmp_path, com)
    assert score_c6_rigid_transport.score(
        scene, tmp_path, ROOT / "scenes").status == "INVALID"


def test_c7_accepts_settled_rest(tmp_path):
    scene = load_scene("c7_supported_settle_v1.json")
    n = 100
    com = np.zeros((n, 3))
    com[:20, 1] = np.linspace(-0.05, 0.0, 20)  # settling transient
    write_metrics(tmp_path, com, edge_med=1e-3, edge_p95=5e-3, bbox=1.0)
    assert score_c7_supported_settle.score(
        scene, tmp_path, ROOT / "scenes").status == "PASS"


def test_c7_rejects_tail_drift_and_shape_blowup(tmp_path):
    scene = load_scene("c7_supported_settle_v1.json")
    n = 100
    com = np.zeros((n, 3))
    com[:, 0] = np.linspace(0.0, 0.2, n)  # creeping drift never settles
    write_metrics(tmp_path / "drift", com, bbox=1.0)
    assert score_c7_supported_settle.score(
        scene, tmp_path / "drift", ROOT / "scenes").status == "FAIL"
    # Over-aggressive threshold fault: sustained contact clamping distorts shape.
    com2 = np.zeros((n, 3))
    write_metrics(tmp_path / "shape", com2, edge_med=0.1, edge_p95=0.3, bbox=1.0)
    assert score_c7_supported_settle.score(
        scene, tmp_path / "shape", ROOT / "scenes").status == "FAIL"


def test_c8_accepts_impact_with_recovery(tmp_path):
    scene = load_scene("c8_impact_recovery_v1.json")
    n = 60
    com = np.zeros((n, 3))
    com[:, 1] = -np.minimum(np.linspace(0, 1.2, n), 0.5)  # falls then floor
    bump = np.exp(-0.5 * ((np.arange(n) - 25) / 4.0) ** 2)
    write_metrics(tmp_path, com, edge_med=0.2 * bump, edge_p95=0.5 * bump,
                  area_med=0.3 * bump, area_p95=0.6 * bump)
    v = score_c8_impact_recovery.score(scene, tmp_path, ROOT / "scenes")
    assert v.status == "PASS"


def test_c8_rejects_no_recovery_and_gates_gentle_runs(tmp_path):
    scene = load_scene("c8_impact_recovery_v1.json")
    n = 60
    com = np.zeros((n, 3))
    com[:, 1] = -np.minimum(np.linspace(0, 1.2, n), 0.5)
    # Wrong-material fault (e.g. sand): deformation never recovers.
    step = 0.4 * (np.arange(n) >= 25)
    write_metrics(tmp_path / "plastic", com, edge_med=step, edge_p95=step,
                  area_med=step, area_p95=np.maximum(step, 0.06))
    assert score_c8_impact_recovery.score(
        scene, tmp_path / "plastic", ROOT / "scenes").status == "FAIL"
    # Too-gentle run: impact gate makes it INVALID, not PASS.
    write_metrics(tmp_path / "gentle", com, edge_med=1e-4, edge_p95=1e-3,
                  area_med=1e-4, area_p95=1e-3)
    assert score_c8_impact_recovery.score(
        scene, tmp_path / "gentle", ROOT / "scenes").status == "INVALID"
