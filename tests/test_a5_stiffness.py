"""Unit tests for the A5 Poisson stiffness scorer.

Known outcomes:
  - A displacement field with exact Poisson contraction must PASS.
  - The same axial ramp with zero lateral contraction must FAIL.
  - A submission whose prescribed face never moves must return INVALID.
"""

import numpy as np

from conftest import load_scene_json, write_submission
from scoring import common
from scoring import score_a5_stiffness as a5

SCENE = load_scene_json("a5_stiffness_v1.json")
FPS = SCENE["simulation"]["output_fps"]
DUR = SCENE["simulation"]["duration_s"]
RAMP_END = SCENE["boundary_conditions"]["prescribed_face"]["ramp_end_s"]
DELTA = SCENE["boundary_conditions"]["prescribed_face"]["displacement_m"][0]
NU = SCENE["material"]["poisson_ratio"]


def _uniaxial_trajectory(scenes_dir, nu_effective):
    """Exact homogeneous uniaxial-stress deformation with the given
    effective Poisson ratio, ramped over RAMP_END then held."""
    rest = np.load(scenes_dir / SCENE["geometry"]["particle_positions_file"])
    t = np.arange(int(round(DUR * FPS)) + 1) / FPS
    ramp = np.clip(t / RAMP_END, 0.0, 1.0)

    x0 = rest[:, 0]
    x_lo = x0.min()
    gauge = x0.max() - x_lo
    eps_ax = DELTA / gauge

    pos = np.repeat(rest[None, :, :], len(t), axis=0)
    # axial: u_x grows linearly from the fixed face
    pos[:, :, 0] += ramp[:, None] * eps_ax * (x0 - x_lo)[None, :]
    # lateral: contraction about the section centroid
    for axis in (1, 2):
        c = rest[:, axis] - rest[:, axis].mean()
        pos[:, :, axis] += ramp[:, None] * (-nu_effective * eps_ax) * c[None, :]
    return pos, t


def test_exact_poisson_contraction_passes(tmp_path, scenes_dir):
    pos, t = _uniaxial_trajectory(scenes_dir, NU)
    sub = write_submission(tmp_path / "sub", pos, t)
    v = a5.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_PASS
    assert v.measured["rel_error"] < 0.02


def test_zero_contraction_fails(tmp_path, scenes_dir):
    pos, t = _uniaxial_trajectory(scenes_dir, 0.0)
    sub = write_submission(tmp_path / "sub", pos, t)
    v = a5.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_FAIL
    assert abs(v.measured["lateral_strain"]) < 1e-6


def test_missing_prescribed_displacement_is_invalid(tmp_path, scenes_dir):
    rest = np.load(scenes_dir / SCENE["geometry"]["particle_positions_file"])
    t = np.arange(int(round(DUR * FPS)) + 1) / FPS
    pos = np.repeat(rest[None, :, :], len(t), axis=0)  # nothing ever moves
    sub = write_submission(tmp_path / "sub", pos, t)
    v = a5.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_INVALID


def test_wrong_poisson_ratio_fails(tmp_path, scenes_dir):
    # Half the scene's Poisson ratio: 50 percent relative error, over the
    # 10 percent tolerance.
    pos, t = _uniaxial_trajectory(scenes_dir, 0.5 * NU)
    sub = write_submission(tmp_path / "sub", pos, t)
    v = a5.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_FAIL
