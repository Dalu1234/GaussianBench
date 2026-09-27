"""Unit tests for the A3 wave speed scorer.

Known outcomes:
  - A step arrival at a known time must be detected within one frame.
  - A synthetic traveling pulse at the reference speed must PASS.
  - The same pulse at 0.7x the reference speed must FAIL.
"""

import numpy as np

from conftest import load_scene_json, write_submission
from scoring import common
from scoring import score_a3_wavespeed as a3

SCENE = load_scene_json("a3_wavespeed_v1.json")
FPS = SCENE["simulation"]["output_fps"]
DUR = SCENE["simulation"]["duration_s"]
AMP = SCENE["boundary_conditions"]["pulse"]["amplitude_m"]
WIDTH = SCENE["boundary_conditions"]["pulse"]["width_s"]
C_REF = float(np.sqrt(SCENE["material"]["youngs_modulus_pa"]
                      / SCENE["material"]["density_kg_m3"]))


def test_step_arrival_detected_within_one_frame():
    t_true = 0.0447
    t = np.arange(int(round(DUR * FPS)) + 1) / FPS
    disp = np.where(t >= t_true, 1e-3, 0.0)
    k = a3.first_arrival_frame(disp, SCENE["pass_criteria"]["noise_floor_m"])
    k_true = int(np.searchsorted(t, t_true))
    assert abs(k - k_true) <= 1


def _traveling_pulse(scenes_dir, c):
    """u_x(x, t) = A sin(pi (t - x/c) / W) inside the pulse window."""
    rest = np.load(scenes_dir / SCENE["geometry"]["particle_positions_file"])
    t = np.arange(int(round(DUR * FPS)) + 1) / FPS
    x = rest[:, 0] - rest[:, 0].min()
    tau = t[:, None] - x[None, :] / c
    u = np.where((tau >= 0) & (tau <= WIDTH),
                 AMP * np.sin(np.pi * np.clip(tau, 0, WIDTH) / WIDTH), 0.0)
    # after the pulse has passed, the bar keeps the pushed displacement of
    # zero (displacement pulse returns to zero), which matches u above
    pos = np.repeat(rest[None, :, :], len(t), axis=0)
    pos[:, :, 0] += u
    return pos, t


def test_correct_speed_passes(tmp_path, scenes_dir):
    pos, t = _traveling_pulse(scenes_dir, C_REF)
    sub = write_submission(tmp_path / "sub", pos, t)
    v = a3.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_PASS
    assert v.measured["rel_error"] < 0.05


def test_slow_speed_fails(tmp_path, scenes_dir):
    pos, t = _traveling_pulse(scenes_dir, 0.7 * C_REF)
    sub = write_submission(tmp_path / "sub", pos, t)
    v = a3.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_FAIL


def test_no_arrival_fails_with_note(tmp_path, scenes_dir):
    rest = np.load(scenes_dir / SCENE["geometry"]["particle_positions_file"])
    t = np.arange(int(round(DUR * FPS)) + 1) / FPS
    pos = np.repeat(rest[None, :, :], len(t), axis=0)  # bar never moves
    sub = write_submission(tmp_path / "sub", pos, t)
    v = a3.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_FAIL
    assert v.measured["arrival_detected"] is False
