"""Unit tests for the A4 melt front scorer and the Stefan reference.

Known outcomes:
  - Front positions generated exactly from the Neumann solution must PASS.
  - A linear-in-time front must FAIL.
  - A submission without a phase column must be rejected with a format error.
  - The Neumann lambda solver must satisfy its own transcendental equation
    and match the small-Stefan-number one-phase asymptote lam ~ sqrt(St/2).
"""

import numpy as np
import pytest

from conftest import load_scene_json, write_submission
from scoring import common
from scoring import score_a4_meltfront as a4
from reference import stefan

SCENE = load_scene_json("a4_meltfront_v1.json")
FPS = SCENE["simulation"]["output_fps"]
DUR = SCENE["simulation"]["duration_s"]
TH = SCENE["material"]["thermal"]


def _prefactor_ref():
    return stefan.front_prefactor(
        TH["conductivity_w_m_k"], SCENE["material"]["density_kg_m3"],
        TH["specific_heat_j_kg_k"], TH["latent_heat_j_kg"],
        SCENE["boundary_conditions"]["hot_floor"]["temperature_k"],
        TH["melt_temperature_k"], TH["initial_temperature_k"])


def _melt_submission(tmp_path, scenes_dir, front_fn):
    """Static column whose phase flips when front_fn(t) passes rest height."""
    rest = np.load(scenes_dir / SCENE["geometry"]["particle_positions_file"])
    t = np.arange(int(round(DUR * FPS)) + 1) / FPS
    z = rest[:, 2] - rest[:, 2].min()
    front = front_fn(t)
    phase = (z[None, :] <= front[:, None]).astype(np.int8)
    pos = np.repeat(rest[None, :, :], len(t), axis=0)
    return write_submission(tmp_path / "sub", pos, t, phase=phase)


def test_neumann_lambda_satisfies_equation():
    from scipy.special import erf, erfc
    st_l, st_s = 1.4, 0.6
    lam = stefan.neumann_lambda(st_l, st_s)
    e = np.exp(lam ** 2)
    residual = (st_l / (e * erf(lam)) - st_s / (e * erfc(lam))
                - lam * np.sqrt(np.pi))
    assert abs(residual) < 1e-10
    assert 0.0 < lam < 2.0


def test_neumann_lambda_one_phase_asymptote():
    # One-phase Stefan (solid initially at melt temperature): for small St,
    # lam -> sqrt(St / 2). Alexiades and Solomon, chapter 2.
    st = 0.02
    lam = stefan.neumann_lambda(st, 0.0)
    assert lam == pytest.approx(np.sqrt(st / 2.0), rel=0.02)


def test_neumann_front_passes(tmp_path, scenes_dir):
    a_ref = _prefactor_ref()
    sub = _melt_submission(tmp_path, scenes_dir,
                           lambda t: a_ref * np.sqrt(t))
    v = a4.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_PASS
    assert v.measured["rel_error"] < 0.10
    assert v.measured["r2_vs_sqrt_t"] > 0.98


def test_linear_front_fails(tmp_path, scenes_dir):
    a_ref = _prefactor_ref()
    # Same final melt height as the Neumann front, but linear in time.
    v_lin = a_ref * np.sqrt(DUR) / DUR
    sub = _melt_submission(tmp_path, scenes_dir, lambda t: v_lin * t)
    v = a4.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_FAIL


def test_missing_phase_column_rejected(tmp_path, scenes_dir):
    rest = np.load(scenes_dir / SCENE["geometry"]["particle_positions_file"])
    t = np.arange(int(round(DUR * FPS)) + 1) / FPS
    pos = np.repeat(rest[None, :, :], len(t), axis=0)
    sub = write_submission(tmp_path / "sub", pos, t)  # no phase column
    with pytest.raises(common.SubmissionError, match="phase"):
        a4.score(SCENE, sub, scenes_dir)


def test_nothing_melts_fails(tmp_path, scenes_dir):
    sub = _melt_submission(tmp_path, scenes_dir,
                           lambda t: np.full_like(t, -1.0))  # front below floor
    v = a4.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_FAIL
    assert v.measured["melted_anything"] is False
