"""Unit tests for the A2 beam frequency scorer.

Known outcomes:
  - A synthetic damped sinusoid at a chosen frequency must be recovered
    within 0.5 percent by the frequency estimator.
  - A trajectory oscillating at the Euler-Bernoulli reference frequency
    must PASS; one at 1.3x the reference must FAIL.
  - A trajectory whose tip deflection exceeds the small-deflection regime
    bound must return INVALID, not FAIL.
"""

import numpy as np
import pytest

from conftest import load_scene_json, write_submission
from scoring import common
from scoring import score_a2_frequency as a2
from reference import beams

SCENE = load_scene_json("a2_beam_frequency_v1.json")
FPS = SCENE["simulation"]["output_fps"]
DUR = SCENE["simulation"]["duration_s"]
L = SCENE["geometry"]["size_m"][0]


def _f_ref():
    m = SCENE["material"]
    Lx, w, h = SCENE["geometry"]["size_m"]
    return beams.cantilever_f1(m["youngs_modulus_pa"], m["density_kg_m3"],
                               Lx, w, h)


def _beam_oscillation(scenes_dir, f_hz, amplitude=0.005, decay=0.05):
    """Whole-beam mode-shaped oscillation in y at the given frequency."""
    rest = np.load(scenes_dir / SCENE["geometry"]["particle_positions_file"])
    t = np.arange(int(round(DUR * FPS)) + 1) / FPS
    s = (rest[:, 0] - rest[:, 0].min()) / L
    shape = beams.mode1_shape(s) / beams.mode1_shape(np.array([1.0]))[0]
    osc = amplitude * np.sin(2 * np.pi * f_hz * t)[:, None] * shape[None, :]
    osc *= np.exp(-decay * t)[:, None]
    pos = np.repeat(rest[None, :, :], len(t), axis=0)
    pos[:, :, 1] += osc
    return pos, t


def test_estimator_recovers_synthetic_frequency():
    # Pure estimator test, no trajectory machinery: damped sinusoid.
    f_true = 1.5
    t = np.arange(int(round(DUR * FPS)) + 1) / FPS
    sig = np.sin(2 * np.pi * f_true * t) * np.exp(-0.05 * t)
    f_est = a2.dominant_frequency(sig - sig.mean(), FPS)
    assert abs(f_est - f_true) / f_true < 0.005


def test_correct_frequency_passes(tmp_path, scenes_dir):
    f_ref = _f_ref()
    pos, t = _beam_oscillation(scenes_dir, f_ref)
    sub = write_submission(tmp_path / "sub", pos, t)
    v = a2.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_PASS
    assert v.measured["rel_error"] < 0.02


def test_wrong_frequency_fails(tmp_path, scenes_dir):
    pos, t = _beam_oscillation(scenes_dir, 1.3 * _f_ref())
    sub = write_submission(tmp_path / "sub", pos, t)
    v = a2.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_FAIL


def test_large_deflection_is_invalid_not_fail(tmp_path, scenes_dir):
    # Amplitude far beyond 5 percent of the beam length.
    pos, t = _beam_oscillation(scenes_dir, _f_ref(), amplitude=0.08)
    sub = write_submission(tmp_path / "sub", pos, t)
    v = a2.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_INVALID


def test_damping_ratio_reported(tmp_path, scenes_dir):
    f_ref = _f_ref()
    decay = 0.10
    pos, t = _beam_oscillation(scenes_dir, f_ref, decay=decay)
    sub = write_submission(tmp_path / "sub", pos, t)
    v = a2.score(SCENE, sub, scenes_dir)
    zeta_expected = decay / (2 * np.pi * f_ref)
    assert v.measured["damping_ratio"] == pytest.approx(zeta_expected, rel=0.2)
