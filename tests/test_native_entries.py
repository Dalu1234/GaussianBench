import json
from pathlib import Path

import numpy as np
import pandas as pd

from scoring import score_a6_freefall, score_c5_triangle_transport


ROOT = Path(__file__).parents[1]


def load_scene(name):
    return json.loads((ROOT / "scenes" / name).read_text())


def test_a6_accepts_freefall_and_rejects_half_gravity(tmp_path):
    scene = load_scene("a6_freefall_v1.json")
    t = np.array([0.0, 0.04, 0.08, 0.12, 0.16])
    for scale, expected in ((1.0, "PASS"), (0.5, "FAIL")):
        out = tmp_path / str(scale)
        out.mkdir()
        pd.DataFrame({
            "frame": np.arange(len(t)), "time": t,
            "com_x": 0.0, "com_y": 0.0,
            "com_z": scale * 0.5 * -9.8 * t**2,
            "com_vx": 0.0, "com_vy": 0.0,
            "com_vz": scale * -9.8 * t,
            "bbox_volume": 1.0,
            "detF_nonpositive_fraction": 0.0,
            "nonfinite_fraction": 0.0,
            "particle_count": 100,
        }).to_csv(out / "body_metrics.csv", index=False)
        assert score_a6_freefall.score(scene, out, ROOT / "scenes").status == expected


def test_c5_accepts_transport_and_rejects_shape_fault(tmp_path):
    scene = load_scene("c5_triangle_shape_transport_v1.json")
    for error, expected in ((1e-4, "PASS"), (0.5, "FAIL")):
        out = tmp_path / str(error)
        out.mkdir()
        pd.DataFrame({
            "frame": [0, 1], "com_x": [0.0, 0.0],
            "com_y": [0.0, -1.0], "com_z": [0.0, 0.0],
            "edge_rel_error_median": [0.0, error],
            "edge_rel_error_p95": [0.0, error],
            "area_rel_error_median": [0.0, error],
            "area_rel_error_p95": [0.0, error],
            "nonfinite_fraction": [0.0, 0.0],
            "triangle_count": [100, 100],
        }).to_csv(out / "triangle_metrics.csv", index=False)
        verdict = score_c5_triangle_transport.score(
            scene, out, ROOT / "scenes")
        assert verdict.status == expected
