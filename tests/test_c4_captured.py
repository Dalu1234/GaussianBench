"""Unit tests for the C4 captured geometry frequency scorer.

Known outcomes:
  - Oscillation of the irregular asset at the measured-section reference
    frequency must PASS.
  - Oscillation at 1.4x the reference must FAIL (outside the 20 percent
    tolerance).
  - The measured section ratio must match the analytical value for a clean
    rectangular section (sanity of the I/A = <y^2> identity).
"""

import numpy as np
import pytest

from conftest import load_scene_json, write_submission
from scoring import common
from scoring import score_c4_captured as c4
from reference import beams

SCENE = load_scene_json("c4_captured_frequency_v1.json")
FPS = SCENE["simulation"]["output_fps"]
DUR = SCENE["simulation"]["duration_s"]


def test_section_ratio_matches_rectangle():
    # For a rectangle of height h, I/A = h^2 / 12. Build a clean lattice
    # and check the estimator, allowing for the discrete-lattice variance
    # ((n^2 - 1)/n^2 correction with n = 10 layers is under 1 percent).
    h = 0.05
    xs = np.linspace(0.0, 0.4, 80)
    ys = (np.arange(10) + 0.5) * (h / 10) - h / 2
    zs = np.zeros(1)
    g = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    ratio = c4.measured_section_ratio(g, 20)
    assert ratio == pytest.approx(h ** 2 / 12.0, rel=0.02)


def _asset_oscillation(scenes_dir, f_hz, amplitude=0.004):
    rest = np.load(scenes_dir / SCENE["geometry"]["particle_positions_file"])
    t = np.arange(int(round(DUR * FPS)) + 1) / FPS
    L = rest[:, 0].max() - rest[:, 0].min()
    s = (rest[:, 0] - rest[:, 0].min()) / L
    shape = beams.mode1_shape(s) / beams.mode1_shape(np.array([1.0]))[0]
    osc = amplitude * np.sin(2 * np.pi * f_hz * t)[:, None] * shape[None, :]
    osc *= np.exp(-0.05 * t)[:, None]
    pos = np.repeat(rest[None, :, :], len(t), axis=0)
    pos[:, :, 1] += osc
    return pos, t


def test_reference_frequency_passes(tmp_path, scenes_dir):
    rest = np.load(scenes_dir / SCENE["geometry"]["particle_positions_file"])
    f_ref = c4.reference_frequency(SCENE, rest)
    pos, t = _asset_oscillation(scenes_dir, f_ref)
    sub = write_submission(tmp_path / "sub", pos, t)
    v = c4.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_PASS
    assert v.measured["rel_error"] < 0.02


def test_wrong_frequency_fails(tmp_path, scenes_dir):
    rest = np.load(scenes_dir / SCENE["geometry"]["particle_positions_file"])
    f_ref = c4.reference_frequency(SCENE, rest)
    pos, t = _asset_oscillation(scenes_dir, 1.4 * f_ref)
    sub = write_submission(tmp_path / "sub", pos, t)
    v = c4.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_FAIL
