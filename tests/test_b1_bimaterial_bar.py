"""Regression tests for the B1 bimaterial-bar scorer."""

import numpy as np

from conftest import load_scene_json, write_submission
from scoring import common
from scoring import score_b1_bimaterial_bar as b1

SCENE = load_scene_json("b1_bimaterial_bar_v1.json")


def _trajectory(scenes_dir, eps_left_scale=1.0, eps_right_scale=1.0):
    rest = np.load(
        scenes_dir / SCENE["geometry"]["particle_positions_file"])
    fps = float(SCENE["simulation"]["output_fps"])
    duration = float(SCENE["simulation"]["duration_s"])
    ramp_end = float(
        SCENE["boundary_conditions"]["prescribed_face"]["ramp_end_s"])
    delta = float(
        SCENE["boundary_conditions"]["prescribed_face"]["displacement_m"][0])
    xi = float(SCENE["material"]["interface_x_m"])
    E1, E2 = [
        float(r["youngs_modulus_pa"]) for r in SCENE["material"]["regions"]]
    x0 = rest[:, 0]
    spacing = float(SCENE["geometry"]["particle_spacing_m"])
    fixed = x0 - x0.min() < 0.75 * spacing
    prescribed = x0.max() - x0 < 0.75 * spacing
    L1 = xi - float(x0[fixed].mean())
    L2 = float(x0[prescribed].mean()) - xi
    sigma = delta / (L1 / E1 + L2 / E2)
    eps1 = sigma / E1 * eps_left_scale
    eps2 = sigma / E2 * eps_right_scale

    u = np.where(x0 < xi, eps1 * (x0 - x0[fixed].mean()),
                 eps1 * L1 + eps2 * (x0 - xi))
    # Preserve the exact commanded displacement on the prescribed layer.
    u[prescribed] = delta
    t = np.arange(int(round(duration * fps)) + 1) / fps
    ramp = np.clip(t / ramp_end, 0.0, 1.0)
    pos = np.repeat(rest[None], len(t), axis=0)
    pos[:, :, 0] += ramp[:, None] * u[None]
    return pos, t


def test_analytical_series_solution_passes(tmp_path, scenes_dir):
    pos, t = _trajectory(scenes_dir)
    sub = write_submission(tmp_path / "sub", pos, t)
    verdict = b1.score(SCENE, sub, scenes_dir)
    assert verdict.status == common.STATUS_PASS
    assert verdict.measured["interface_traction_jump_rel"] < 1e-6


def test_missing_strain_jump_fails(tmp_path, scenes_dir):
    # Force both halves to the soft-region strain: displacement is still
    # prescribed, but inferred interface traction differs by 4x.
    pos, t = _trajectory(scenes_dir, eps_left_scale=1.0,
                         eps_right_scale=4.0)
    sub = write_submission(tmp_path / "sub", pos, t)
    verdict = b1.score(SCENE, sub, scenes_dir)
    assert verdict.status == common.STATUS_FAIL


def test_missing_end_displacement_is_invalid(tmp_path, scenes_dir):
    rest = np.load(
        scenes_dir / SCENE["geometry"]["particle_positions_file"])
    fps = float(SCENE["simulation"]["output_fps"])
    duration = float(SCENE["simulation"]["duration_s"])
    t = np.arange(int(round(duration * fps)) + 1) / fps
    pos = np.repeat(rest[None], len(t), axis=0)
    sub = write_submission(tmp_path / "sub", pos, t)
    verdict = b1.score(SCENE, sub, scenes_dir)
    assert verdict.status == common.STATUS_INVALID
