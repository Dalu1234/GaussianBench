"""Unit tests for the C2 blob health scorer.

Known outcomes:
  - Healthy sheared covariances (correct push-forward of isotropic rest
    Gaussians through gamma = 1 simple shear, covariance condition number
    about 6.85)
    must PASS.
  - One particle driven to a degenerate covariance must FAIL.
  - A frozen covariance field must fail the exact peak-shear transport check.
  - The eigenvalue history plot must be written.
"""

import numpy as np

from conftest import load_scene_json, write_covariances, write_submission
from scoring import common
from scoring import score_c2_blobhealth as c2

SCENE = load_scene_json("c2_blobhealth_shear_v1.json")
FPS = SCENE["simulation"]["output_fps"]
DUR = SCENE["simulation"]["duration_s"]
GAMMA = SCENE["boundary_conditions"]["shear_drive"]["gamma_peak"]


def _shear_submission(tmp_path, scenes_dir, degenerate_particle=None,
                      freeze_covariance=False):
    rest = np.load(scenes_dir / SCENE["geometry"]["particle_positions_file"])
    n = len(rest)
    t = np.arange(int(round(DUR * FPS)) + 1) / FPS
    gamma = GAMMA * (1.0 - np.abs(2.0 * t / t[-1] - 1.0))  # triangle profile

    pos = np.repeat(rest[None, :, :], len(t), axis=0)
    pos[:, :, 0] += gamma[:, None] * rest[None, :, 1]

    sigma0 = 1e-6
    covs = np.empty((len(t), n, 3, 3))
    Fs = np.empty((len(t), n, 3, 3))
    for k, g in enumerate(gamma):
        F = np.eye(3)
        F[0, 1] = g
        covs[k] = (F @ (sigma0 * np.eye(3)) @ F.T)[None, :, :]
        Fs[k] = F
    if freeze_covariance:
        covs[:] = covs[0]
    if degenerate_particle is not None:
        # Crush one particle's smallest eigenvalue progressively to zero.
        scale = np.linspace(1.0, 1e-6, len(t))
        covs[:, degenerate_particle, 2, 2] = sigma0 * scale

    sub = write_submission(tmp_path / "sub", pos, t, deformation_gradients=Fs)
    write_covariances(sub, np.arange(len(t)), covs)
    return sub


def test_healthy_shear_passes(tmp_path, scenes_dir):
    sub = _shear_submission(tmp_path, scenes_dir)
    v = c2.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_PASS
    assert v.measured["peak_kinematic_median_rel_frobenius_error"] < 1e-12
    assert v.measured["peak_transport_median_rel_frobenius_error"] < 1e-12
    assert v.measured["min_eigenvalue_ratio"] > 0.3
    assert v.measured["degenerate_fraction"] == 0.0


def test_degenerate_particle_fails(tmp_path, scenes_dir):
    sub = _shear_submission(tmp_path, scenes_dir, degenerate_particle=17)
    v = c2.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_FAIL
    assert v.measured["worst_particle_id"] == 17
    assert v.measured["min_eigenvalue_ratio"] < 1e-3


def test_frozen_covariance_fails_exact_transport(tmp_path, scenes_dir):
    sub = _shear_submission(tmp_path, scenes_dir, freeze_covariance=True)
    v = c2.score(SCENE, sub, scenes_dir)
    assert v.status == common.STATUS_FAIL
    assert v.measured["peak_transport_median_rel_frobenius_error"] > 0.1


def test_plot_written(tmp_path, scenes_dir):
    sub = _shear_submission(tmp_path, scenes_dir)
    c2.score(SCENE, sub, scenes_dir)
    assert (sub / "c2_worst_particle_eigenvalues.png").exists()
