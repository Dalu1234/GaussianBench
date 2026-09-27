"""Unit tests for the BenchRecorder writer, isolated from the simulator.

Feeds synthetic numpy state for 3 frames and 5 particles, then asserts the
parquet round-trips with correct dtypes, column names, and stable
particle_ids, and that meta.json is valid JSON with all required keys.

Run from the Representation repo root:
    python -m pytest export/test_benchexport.py -q
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from export.benchexport import BenchRecorder

N, F = 5, 3


@pytest.fixture
def scene_json(tmp_path) -> Path:
    scene = {
        "scene_id": "writer_test_v0",
        "test_class": "code_verification",
        "tier": "A",
        "simulation": {"duration_s": 0.5, "output_fps": 4, "gravity": [0, 0, 0]},
        "pass_criteria": {},
    }
    p = tmp_path / "scene.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(scene, f)
    return p


def _state(k):
    rng = np.random.default_rng(k)
    return {
        "positions": rng.normal(0.0, 1.0, (N, 3)).astype(np.float32),
        "velocities": rng.normal(0.0, 1.0, (N, 3)).astype(np.float32),
        "mass": np.full(N, 0.2, dtype=np.float32),
        "phase": np.zeros(N, dtype=np.int8),
        "covariances": np.repeat(np.eye(3)[None] * 1e-6, N, axis=0),
    }


def test_trajectory_roundtrip(tmp_path, scene_json):
    rec = BenchRecorder(tmp_path / "sub", scene_json,
                        record_phase=True, record_covariances=True)
    states = [_state(k) for k in range(F)]
    for k, st in enumerate(states):
        rec.on_frame(st, k, k * 0.25)
    out = rec.finalize()

    df = pd.read_parquet(out / "trajectory.parquet")
    assert list(df.columns) == ["frame", "time", "particle_id",
                                "x", "y", "z", "vx", "vy", "vz",
                                "mass", "phase"]
    assert len(df) == N * F
    assert df["frame"].dtype == np.int64
    assert df["particle_id"].dtype == np.int64
    assert df["time"].dtype == np.float64
    for c in ("x", "y", "z", "vx", "vy", "vz", "mass"):
        assert df[c].dtype == np.float32
    # stable particle ids, same set every frame
    for k in range(F):
        ids = df.loc[df["frame"] == k, "particle_id"].to_numpy()
        assert np.array_equal(np.sort(ids), np.arange(N))
    # positions survive exactly (float32 in, float32 out)
    got = df[df["frame"] == 1].sort_values("particle_id")[["x", "y", "z"]]
    np.testing.assert_array_equal(got.to_numpy(),
                                  states[1]["positions"])
    # times per frame
    assert df[df["frame"] == 2]["time"].unique().tolist() == [0.5]


def test_covariances_roundtrip(tmp_path, scene_json):
    rec = BenchRecorder(tmp_path / "sub", scene_json,
                        record_covariances=True)
    for k in range(F):
        rec.on_frame(_state(k), k, k * 0.25)
    out = rec.finalize()
    cf = pd.read_parquet(out / "covariances.parquet")
    assert list(cf.columns) == ["frame", "particle_id",
                                "sxx", "sxy", "sxz", "syy", "syz", "szz"]
    assert len(cf) == N * F
    row = cf[(cf["frame"] == 0) & (cf["particle_id"] == 0)].iloc[0]
    assert row["sxx"] == pytest.approx(1e-6)
    assert row["sxy"] == 0.0


def test_meta_json_complete(tmp_path, scene_json):
    rec = BenchRecorder(tmp_path / "sub", scene_json,
                        notes=["solver=UL-MLS-MPM", "test note"])
    for k in range(F):
        rec.on_frame(_state(k), k, k * 0.25)
    out = rec.finalize(extra_meta={"seed": 0})
    with open(out / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    for key in ("system_name", "system_version", "spec_version",
                "columns_present", "wall_clock_seconds", "hardware", "notes"):
        assert key in meta, f"meta.json missing {key}"
    assert meta["system_name"] == "GaussianFlesh"
    assert "UL-MLS-MPM" in meta["notes"]
    assert set(meta["columns_present"]) == {"vx", "vy", "vz", "mass"}


def test_offset_transform_inverted(tmp_path, scene_json):
    offset = [0.0, 0.7, 0.0]
    rec = BenchRecorder(tmp_path / "sub", scene_json,
                        scene_to_sim={"offset": offset, "permutation": None})
    st = _state(0)
    sim_pos = st["positions"] + np.asarray(offset, dtype=np.float32)
    rec.on_frame({**st, "positions": sim_pos}, 0, 0.0)
    out = rec.finalize()
    df = pd.read_parquet(out / "trajectory.parquet")
    got = df.sort_values("particle_id")[["x", "y", "z"]].to_numpy()
    np.testing.assert_allclose(got, st["positions"], atol=1e-6)
    with open(out / "meta.json", "r", encoding="utf-8") as f:
        assert "transformed frame" in json.load(f)["notes"]


def test_permutation_transform_inverted(tmp_path, scene_json):
    # scene z-up to sim y-up: sim = P @ scene with P = Rx(+90):
    # (x, y, z)_scene -> (x, z, -y)_sim
    P = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    rec = BenchRecorder(tmp_path / "sub", scene_json,
                        record_covariances=True,
                        scene_to_sim={"offset": [0, 0, 0], "permutation": P})
    st = _state(0)
    Pm = np.asarray(P, dtype=np.float64)
    sim_pos = st["positions"] @ Pm.T
    sim_cov = np.einsum("ij,njk,lk->nil", Pm, st["covariances"], Pm)
    rec.on_frame({**st, "positions": sim_pos, "covariances": sim_cov,
                  "velocities": st["velocities"] @ Pm.T}, 0, 0.0)
    out = rec.finalize()
    df = pd.read_parquet(out / "trajectory.parquet")
    got = df.sort_values("particle_id")[["x", "y", "z"]].to_numpy()
    np.testing.assert_allclose(got, st["positions"], atol=1e-6)
    cf = pd.read_parquet(out / "covariances.parquet").sort_values("particle_id")
    np.testing.assert_allclose(cf["sxy"].to_numpy(),
                               st["covariances"][:, 0, 1], atol=1e-9)


def test_particle_count_change_rejected(tmp_path, scene_json):
    rec = BenchRecorder(tmp_path / "sub", scene_json)
    rec.on_frame(_state(0), 0, 0.0)
    bad = _state(1)
    bad["positions"] = bad["positions"][:3]
    bad["velocities"] = bad["velocities"][:3]
    with pytest.raises(ValueError, match="constant"):
        rec.on_frame(bad, 1, 0.25)
