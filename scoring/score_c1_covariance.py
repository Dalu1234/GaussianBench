"""C1 covariance push-forward correctness (test_class: code_verification).

Under an affine deformation x -> F x, a Gaussian with rest covariance
Sigma_0 must end with world covariance F Sigma_0 F^T. This scorer separates
two contracts that must not be conflated: (1) kinematic realization compares
a position-fitted local F_i with prescribed F_target; (2) pure transport
compares the submitted covariance with the entrant-native F_i Sigma_i(0) F_i^T. The latter asks
whether covariance follows the deformation that the entrant actually made,
independent of how closely that deformation matches the prescribed scene.

Particles within pass_criteria.boundary_exclusion_spacings particle spacings
of any prescribed face are excluded, because interior homogeneity is only
approximate there (the scene provenance says the same). Metrics are the
median and 95th percentile of the relative Frobenius error across the
remaining particles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import common


def score(scene: dict, results_dir: Path, scenes_dir: Path) -> common.Verdict:
    traj = common.load_trajectory(results_dir)
    rest = common.load_scene_positions(scene, scenes_dir)
    common.check_initial_condition(traj, rest)
    common.check_output_fps(traj, scene)

    crit = scene["pass_criteria"]
    F_target = np.asarray(scene["boundary_conditions"]["all_faces"]["F_target"],
                          dtype=np.float64)

    final_frame = traj.n_frames - 1
    covs = common.load_covariances(results_dir, frames=[0, final_frame])
    cov0, cov1 = covs[0], covs[final_frame]
    native_F = common.load_deformation_gradients(
        results_dir, frames=[0, final_frame])[final_frame]
    if cov0.shape[0] != traj.n_particles:
        raise common.SubmissionError(
            f"covariances.parquet has {cov0.shape[0]} particles per frame but "
            f"the trajectory has {traj.n_particles}.")

    # Exclusion zone: within k spacings of any face of the rest bounding box.
    k = float(crit["boundary_exclusion_spacings"])
    h = float(scene["geometry"]["particle_spacing_m"])
    lo = rest.min(axis=0) + k * h
    hi = rest.max(axis=0) - k * h
    core = np.all((rest > lo) & (rest < hi), axis=1)
    n_core = int(core.sum())
    if n_core == 0:
        raise common.SubmissionError(
            "Boundary exclusion removed every particle; geometry too small "
            "for the declared exclusion zone.")

    ids, F_fit = common.fit_local_deformation_gradients(
        rest, traj.positions[final_frame], core)
    kin_rel = (np.linalg.norm(F_fit - F_target[None], axis=(1, 2))
               / np.linalg.norm(F_target))
    predicted_observed = np.einsum(
        "nij,njk,nlk->nil", native_F[ids], cov0[ids], native_F[ids])
    transport_rel = (
        np.linalg.norm(cov1[ids] - predicted_observed, axis=(1, 2))
        / np.linalg.norm(predicted_observed, axis=(1, 2)))

    kin_med = float(np.median(kin_rel))
    kin_p95 = float(np.percentile(kin_rel, 95))
    tr_med = float(np.median(transport_rel))
    tr_p95 = float(np.percentile(transport_rel, 95))
    kin_ok = (kin_med < crit["kinematic_median_max"]
              and kin_p95 < crit["kinematic_p95_max"])
    transport_ok = (tr_med < crit["transport_median_max"]
                    and tr_p95 < crit["transport_p95_max"])
    ok = kin_ok and transport_ok

    return common.verdict_from_scene(
        scene,
        common.STATUS_PASS if ok else common.STATUS_FAIL,
        measured={"kinematic_median_rel_frobenius_error": kin_med,
                  "kinematic_p95_rel_frobenius_error": kin_p95,
                  "transport_median_rel_frobenius_error": tr_med,
                  "transport_p95_rel_frobenius_error": tr_p95,
                  "kinematic_status": "PASS" if kin_ok else "FAIL",
                  "transport_status": "PASS" if transport_ok else "FAIL",
                  "n_particles_scored": n_core},
        reference={"kinematic_target": "F_fit from positions vs F_target",
                   "transport_map": "F_native Sigma_0 F_native^T",
                   "F_target_diag": np.diag(F_target).tolist()},
        tolerance={"kinematic_median_max": crit["kinematic_median_max"],
                   "kinematic_p95_max": crit["kinematic_p95_max"],
                   "transport_median_max": crit["transport_median_max"],
                   "transport_p95_max": crit["transport_p95_max"]},
        notes=[f"Excluded {traj.n_particles - n_core} particles within "
               f"{k:g} spacings of a prescribed face; fitted and scored "
               f"local F on {n_core} particles; transport uses the entrant's "
               f"exported internal F. Kinematics "
               f"{'pass' if kin_ok else 'fail'}; pure covariance transport "
               f"{'passes' if transport_ok else 'fails'}."])
