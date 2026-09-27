"""Unit tests for the C1 covariance push-forward scorer.

Known outcomes:
  - Covariances generated exactly as F Sigma_0 F^T must PASS at machine
    precision.
  - Final covariances left at their rest values (no push-forward) must FAIL.
"""

import numpy as np

from conftest import load_scene_json, write_covariances, write_submission
from scoring import common
from scoring import score_c1_covariance as c1

SCENE = load_scene_json("c1_covariance_v1.json")
FPS = SCENE["simulation"]["output_fps"]
DUR = SCENE["simulation"]["duration_s"]
F_TARGET = np.asarray(SCENE["boundary_conditions"]["all_faces"]["F_target"])


def _affine_trajectory(scenes_dir):
    rest = np.load(scenes_dir / SCENE["geometry"]["particle_positions_file"])
    t = np.arange(int(round(DUR * FPS)) + 1) / FPS
    ramp = t / t[-1]
    pos = np.empty((len(t), len(rest), 3))
    Fs = np.empty((len(t), len(rest), 3, 3))
    for k, r in enumerate(ramp):
        F = np.eye(3) + r * (F_TARGET - np.eye(3))
        pos[k] = rest @ F.T
        Fs[k] = F
    return pos, t, rest, Fs


def _rest_covariances(n, seed=7):
    """Random SPD rest covariances at the particle length scale."""
    rng = np.random.default_rng(seed)
    A = rng.normal(0.0, 1.0, (n, 3, 3))
    covs = np.einsum("nij,nkj->nik", A, A) * 1e-6
    covs += np.eye(3)[None] * 1e-6
    return covs


def test_exact_pushforward_passes(tmp_path, scenes_dir):
    pos, t, rest, Fs = _affine_trajectory(scenes_dir)
    sub = write_submission(tmp_path / "sub", pos, t, deformation_gradients=Fs)
    cov0 = _rest_covariances(len(rest))
    cov1 = np.einsum("ij,njk,lk->nil", F_TARGET, cov0, F_TARGET)
    write_covariances(sub, [0, len(t) - 1], np.stack([cov0, cov1]))
    v = c1.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_PASS
    assert v.measured["kinematic_status"] == common.STATUS_PASS
    assert v.measured["transport_status"] == common.STATUS_PASS
    assert v.measured["transport_median_rel_frobenius_error"] < 1e-12


def test_missing_pushforward_fails(tmp_path, scenes_dir):
    pos, t, rest, Fs = _affine_trajectory(scenes_dir)
    sub = write_submission(tmp_path / "sub", pos, t, deformation_gradients=Fs)
    cov0 = _rest_covariances(len(rest))
    # Final covariances identical to rest: the system never deformed them.
    write_covariances(sub, [0, len(t) - 1], np.stack([cov0, cov0]))
    v = c1.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_FAIL
    assert v.measured["transport_status"] == "FAIL"


def test_missing_covariance_file_is_submission_error(tmp_path, scenes_dir):
    pos, t, rest, Fs = _affine_trajectory(scenes_dir)
    sub = write_submission(tmp_path / "sub", pos, t)
    import pytest
    with pytest.raises(common.SubmissionError, match="covariances.parquet"):
        c1.score(SCENE, sub, scenes_dir)
