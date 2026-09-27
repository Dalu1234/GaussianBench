"""Unit tests for the conservation scorer on A1a (in-place, pure spin).

Known outcomes, all constructed analytically:
  - Exact rigid spin (zero net momentum) must PASS at the tight tiers.
  - Rotational velocities scaled by (1 + 1e-2 t) inject angular momentum
    drift and must FAIL on the L metric.
  - A position-only submission (derived velocities) of the conserving spin
    must PASS.
  - A trajectory that conserves p and L exactly but grows kinetic energy
    monotonically (zero-sum pairwise radial kicks) must FAIL, because A1a
    sets energy_monotone_growth_fails.
"""

import numpy as np

from conftest import load_scene_json, rigid_trajectory, write_submission
from scoring import common
from scoring import score_a1_conservation as a1

SCENE = load_scene_json("a1a_conservation_inplace_v1.json")
FPS = SCENE["simulation"]["output_fps"]
OMEGA_Z = SCENE["initial_conditions"]["angular_velocity_rad_s"][2]
V0 = np.asarray(SCENE["initial_conditions"]["linear_velocity_m_s"])


def _times(duration=2.0):
    return np.arange(int(round(duration * FPS)) + 1) / FPS


def _spin(scenes_dir):
    x0 = np.load(scenes_dir / SCENE["geometry"]["particle_positions_file"])
    t = _times()
    assert np.all(V0 == 0.0), "A1a must prescribe zero linear velocity"
    pos, vel = rigid_trajectory(x0, V0, OMEGA_Z, t)
    return pos, vel, t


def test_pure_spin_passes(tmp_path, scenes_dir):
    pos, vel, t = _spin(scenes_dir)
    sub = write_submission(tmp_path / "sub", pos, t, velocities=vel)
    v = a1.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_PASS
    assert v.measured["linear_momentum_rel_drift"] < 1e-9
    assert v.measured["angular_momentum_rel_drift"] < 1e-9


def test_injected_angular_drift_fails(tmp_path, scenes_dir):
    pos, vel, t = _spin(scenes_dir)
    vel_bad = vel * (1.0 + 1e-2 * t)[:, None, None]
    sub = write_submission(tmp_path / "sub", pos, t, velocities=vel_bad)
    v = a1.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_FAIL
    assert v.measured["angular_momentum_rel_drift"] > 1e-3


def test_derived_velocities_pass(tmp_path, scenes_dir):
    pos, _, t = _spin(scenes_dir)
    sub = write_submission(tmp_path / "sub", pos, t)  # positions only
    v = a1.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_PASS
    assert v.tolerance["velocities_derived"] is True


def test_monotone_energy_growth_fails(tmp_path, scenes_dir):
    # Zero-sum pairwise kicks along the pair separation conserve p exactly
    # (opposite forces) and L exactly (the kick is parallel to r_i - r_j,
    # so (r_i - r_j) x u = 0), while kinetic energy grows monotonically.
    pos, vel, t = _spin(scenes_dir)
    n = pos.shape[1] - (pos.shape[1] % 2)
    i, j = np.arange(0, n, 2), np.arange(1, n, 2)
    ke0 = 0.5 * (vel[0] ** 2).sum()
    amp = 0.5 * np.sqrt(ke0 / n)  # comfortably above the 1 percent flag
    vel_bad = vel.copy()
    for k, tk in enumerate(t):
        d = pos[k, i] - pos[k, j]
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        vel_bad[k, i] += amp * tk * d
        vel_bad[k, j] -= amp * tk * d
    sub = write_submission(tmp_path / "sub", pos, t, velocities=vel_bad)
    v = a1.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_FAIL
    assert v.measured["linear_momentum_rel_drift"] < 1e-6
    assert v.measured["angular_momentum_rel_drift"] < 1e-3
    assert v.measured["kinetic_energy_rel_change_final"] > 0.01
    assert any("FAIL per this scene" in note for note in v.notes)
