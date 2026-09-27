"""Unit tests for the conservation scorer on A1b (drifting + declaration).

Known outcomes, all constructed analytically:
  - Conserving translating-and-spinning trajectory with a valid
    meta.json notes.domain_handling = "fixed": PASS at the looser tiers.
  - Same trajectory with meta.json missing the declaration: INVALID.
  - Declaration "co_moving": PASS, and the verdict echoes it in
    measured["declared_domain_handling"].
  - Declaration outside the allowed set: INVALID.
  - Momentum drift beyond even the looser 1e-3 tier still FAILS with a
    valid declaration (the declaration is not a pass ticket).
"""

import numpy as np

from conftest import load_scene_json, rigid_trajectory, write_submission
from scoring import common
from scoring import score_a1_conservation as a1

SCENE = load_scene_json("a1b_conservation_drifting_v1.json")
FPS = SCENE["simulation"]["output_fps"]
V0 = np.asarray(SCENE["initial_conditions"]["linear_velocity_m_s"])
OMEGA_Z = SCENE["initial_conditions"]["angular_velocity_rad_s"][2]

DECLARED = {"notes": {"domain_handling": "fixed",
                      "text": "synthetic trajectory from the unit tests"}}


def _drifting(scenes_dir, duration=2.0):
    x0 = np.load(scenes_dir / SCENE["geometry"]["particle_positions_file"])
    t = np.arange(int(round(duration * FPS)) + 1) / FPS
    pos, vel = rigid_trajectory(x0, V0, OMEGA_Z, t)
    return pos, vel, t


def test_declared_conserving_passes(tmp_path, scenes_dir):
    pos, vel, t = _drifting(scenes_dir)
    sub = write_submission(tmp_path / "sub", pos, t, velocities=vel,
                           meta=DECLARED)
    v = a1.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_PASS
    assert v.tolerance["linear_momentum_rel_drift"] == 1e-3
    assert v.measured["declared_domain_handling"] == "fixed"


def test_missing_declaration_is_invalid(tmp_path, scenes_dir):
    pos, vel, t = _drifting(scenes_dir)
    # default conftest meta: notes is a plain string, no declaration
    sub = write_submission(tmp_path / "sub", pos, t, velocities=vel)
    v = a1.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_INVALID
    assert any("domain_handling" in note for note in v.notes)


def test_co_moving_declaration_echoed(tmp_path, scenes_dir):
    pos, vel, t = _drifting(scenes_dir)
    meta = {"notes": {"domain_handling": "co_moving", "text": "synthetic"}}
    sub = write_submission(tmp_path / "sub", pos, t, velocities=vel,
                           meta=meta)
    v = a1.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_PASS
    assert v.measured["declared_domain_handling"] == "co_moving"


def test_disallowed_declaration_is_invalid(tmp_path, scenes_dir):
    pos, vel, t = _drifting(scenes_dir)
    meta = {"notes": {"domain_handling": "magic_grid"}}
    sub = write_submission(tmp_path / "sub", pos, t, velocities=vel,
                           meta=meta)
    v = a1.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_INVALID
    assert any("allowed set" in note for note in v.notes)


def test_allowed_p_drift_does_not_contaminate_L(tmp_path, scenes_dir):
    # Regression for the v0.2.1 erratum: a constant velocity offset ramping
    # to HALF the allowed linear tolerance (a legal p drift) must not fail
    # the angular metric. Under the old L-about-initial-cm definition, this
    # trajectory showed an apparent L drift of order 1 through the
    # M (x_cm - x_cm0) x v_cm cross term with the drift lever arm, 100x its
    # tolerance, so the L metric was measuring the p drift twice.
    pos, vel, t = _drifting(scenes_dir, duration=4.0)
    p0_mag = np.linalg.norm(V0)  # per unit mass, equal masses
    dv = 0.5 * 1e-3 * p0_mag * np.array([0.0, 0.0, 1.0])
    ramp = (t / t[-1])[:, None, None]
    vel_bad = vel + ramp * dv[None, None, :]
    # positions consistent with the perturbed velocities (integrate the ramp)
    pos_bad = pos + (0.5 * t[:, None] ** 2 / t[-1] * dv[None, :])[:, None, :]
    # keep frame 0 exact
    pos_bad[0] = pos[0]
    sub = write_submission(tmp_path / "sub", pos_bad, t, velocities=vel_bad,
                           meta=DECLARED)
    v = a1.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_PASS
    assert v.measured["linear_momentum_rel_drift"] < 1e-3
    assert v.measured["angular_momentum_rel_drift"] < 1e-6


def test_declaration_is_not_a_pass_ticket(tmp_path, scenes_dir):
    pos, vel, t = _drifting(scenes_dir)
    vel_bad = vel * (1.0 + 5e-3 * t)[:, None, None]  # ~1 percent drift
    sub = write_submission(tmp_path / "sub", pos, t, velocities=vel_bad,
                           meta=DECLARED)
    v = a1.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_FAIL
    assert v.measured["linear_momentum_rel_drift"] > 1e-3
