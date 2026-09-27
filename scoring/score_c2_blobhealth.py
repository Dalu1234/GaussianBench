"""C2 covariance transport and blob health under violent deformation.

Positions can look plausible while the Gaussians themselves die: covariances
collapsing to needles or pancakes (an eigenvalue heading to zero) or blowing
up (huge condition number) destroy rendering quality even when the centroid
motion is fine. This scorer watches the covariance eigenvalue spectrum of
every particle across the whole run.

Metrics (thresholds come from the scene pass_criteria):
  - peak_kinematic_*: local position-fitted F_i versus prescribed peak shear.
  - peak_transport_*: covariance push-forward error against each entrant's
    exported internal F_i at peak shear. This isolates transport from the
    position-fit mismatch.
  - min_eigenvalue_ratio: the global minimum eigenvalue over all particles
    and frames, relative to the median eigenvalue of the initial frame.
    Guards against collapse.
  - degenerate_fraction: fraction of particle-frames whose condition number
    lambda_max / lambda_min exceeds condition_number_max. Guards against
    anisotropic blowup.

The scorer also saves a PNG of the worst particle's eigenvalue history
(matplotlib, Agg backend) into the submission directory for the report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import common


def eigenvalue_traces(covs_by_frame: dict[int, np.ndarray]):
    """Return (frames, eigs) with eigs of shape (F, N, 3), ascending order."""
    frames = sorted(covs_by_frame)
    eigs = np.stack([np.linalg.eigvalsh(covs_by_frame[f]) for f in frames])
    return np.asarray(frames), eigs


def score(scene: dict, results_dir: Path, scenes_dir: Path) -> common.Verdict:
    traj = common.load_trajectory(results_dir)
    rest = common.load_scene_positions(scene, scenes_dir)
    common.check_initial_condition(traj, rest)
    common.check_output_fps(traj, scene)

    covs = common.load_covariances(results_dir)
    if 0 not in covs:
        raise common.SubmissionError(
            "covariances.parquet must include frame 0 (the rest state).")
    if len(covs) < traj.n_frames:
        raise common.SubmissionError(
            f"C2 requires covariances for every output frame: trajectory has "
            f"{traj.n_frames} frames, covariances cover {len(covs)}.")

    crit = scene["pass_criteria"]
    frames, eigs = eigenvalue_traces(covs)   # (F, N, 3) ascending

    peak_frame = int(crit["peak_frame"])
    if peak_frame not in covs:
        raise common.SubmissionError(
            f"C2 requires the prescribed peak-shear frame {peak_frame}.")
    gamma_peak = float(scene["boundary_conditions"]["shear_drive"]["gamma_peak"])
    F_peak = np.eye(3)
    F_peak[0, 1] = gamma_peak
    native_F = common.load_deformation_gradients(
        results_dir, frames=[0, peak_frame])[peak_frame]
    # Prescribed-face particles and their immediate stencil neighborhood do
    # not realize a homogeneous interior deformation.  As in C1, exclude a
    # frozen boundary band from the exact transport statistic while keeping
    # every particle and frame in the separate health audit below.
    k = float(crit["peak_transport_boundary_exclusion_spacings"])
    h = float(scene["geometry"]["particle_spacing_m"])
    lo = rest.min(axis=0) + k * h
    hi = rest.max(axis=0) - k * h
    core = np.all((rest > lo) & (rest < hi), axis=1)
    if not np.any(core):
        raise common.SubmissionError(
            "C2 peak-transport boundary exclusion removed every particle.")
    ids, F_fit = common.fit_local_deformation_gradients(
        rest, traj.positions[peak_frame], core)
    kin_rel = (np.linalg.norm(F_fit - F_peak[None], axis=(1, 2))
               / np.linalg.norm(F_peak))
    predicted_observed = np.einsum(
        "nij,njk,nlk->nil", native_F[ids], covs[0][ids], native_F[ids])
    transport_rel = (
        np.linalg.norm(covs[peak_frame][ids] - predicted_observed,
                       axis=(1, 2))
        / np.linalg.norm(predicted_observed, axis=(1, 2)))
    kin_med = float(np.median(kin_rel))
    kin_p95 = float(np.percentile(kin_rel, 95))
    peak_med = float(np.median(transport_rel))
    peak_p95 = float(np.percentile(transport_rel, 95))

    lam_min = eigs[..., 0]
    lam_max = eigs[..., 2]
    init_median = float(np.median(eigs[0]))
    if init_median <= 0:
        raise common.SubmissionError(
            "Initial covariances are not positive definite "
            "(median eigenvalue <= 0).")

    min_ratio = float(lam_min.min() / init_median)
    with np.errstate(divide="ignore"):
        cond = np.where(lam_min > 0, lam_max / np.maximum(lam_min, 1e-300),
                        np.inf)
    degenerate = cond > crit["condition_number_max"]
    degen_frac = float(degenerate.mean())

    kin_ok = (kin_med <= crit["peak_kinematic_median_rel_frobenius_max"]
              and kin_p95 <= crit["peak_kinematic_p95_rel_frobenius_max"])
    transport_ok = (
          peak_med <= crit["peak_transport_median_rel_frobenius_max"]
          and peak_p95 <= crit["peak_transport_p95_rel_frobenius_max"])
    health_ok = (min_ratio >= crit["min_eigenvalue_ratio"]
                 and degen_frac < crit["degenerate_fraction_max"])
    if not kin_ok:
        status = common.STATUS_INVALID
    elif transport_ok and health_ok:
        status = common.STATUS_PASS
    else:
        status = common.STATUS_FAIL

    # Worst particle: the one that attains the global minimum eigenvalue.
    worst = int(np.unravel_index(np.argmin(lam_min), lam_min.shape)[1])
    times = traj.times[frames] if len(traj.times) > frames.max() else frames
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, label in enumerate(("lambda_min", "lambda_mid", "lambda_max")):
        ax.semilogy(times, eigs[:, worst, j], label=label)
    ax.axhline(init_median * crit["min_eigenvalue_ratio"], ls="--", c="r",
               label="collapse threshold")
    ax.set_xlabel("time [s]" if len(traj.times) > frames.max() else "frame")
    ax.set_ylabel("covariance eigenvalue [m^2]")
    ax.set_title(f"{scene['scene_id']}: worst particle (id {worst})")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    plot_name = "c2_worst_particle_eigenvalues.png"
    fig.savefig(results_dir / plot_name, dpi=110)
    plt.close(fig)

    return common.verdict_from_scene(
        scene,
        status,
        measured={"peak_kinematic_median_rel_frobenius_error": kin_med,
                  "peak_kinematic_p95_rel_frobenius_error": kin_p95,
                  "peak_transport_median_rel_frobenius_error": peak_med,
                  "peak_transport_p95_rel_frobenius_error": peak_p95,
                  "kinematic_status": "PASS" if kin_ok else "FAIL",
                  "transport_status": "PASS" if transport_ok else "FAIL",
                  "health_status": "PASS" if health_ok else "FAIL",
                  "min_eigenvalue_ratio": min_ratio,
                  "degenerate_fraction": degen_frac,
                  "worst_particle_id": worst},
        reference={"kinematic_target": "F_fit from positions vs F_peak",
                   "peak_covariance_map": "F_native Sigma_0 F_native^T",
                   "peak_gamma": gamma_peak,
                   "healthy_min_eigenvalue_ratio": 1.0,
                   "healthy_degenerate_fraction": 0.0},
        tolerance={"peak_kinematic_median_rel_frobenius_max": crit["peak_kinematic_median_rel_frobenius_max"],
                   "peak_kinematic_p95_rel_frobenius_max": crit["peak_kinematic_p95_rel_frobenius_max"],
                   "peak_transport_median_rel_frobenius_max": crit["peak_transport_median_rel_frobenius_max"],
                   "peak_transport_p95_rel_frobenius_max": crit["peak_transport_p95_rel_frobenius_max"],
                   "peak_transport_boundary_exclusion_spacings": k,
                   "min_eigenvalue_ratio": crit["min_eigenvalue_ratio"],
                   "condition_number_max": crit["condition_number_max"],
                   "degenerate_fraction_max": crit["degenerate_fraction_max"]},
        notes=[f"Peak frame {peak_frame}: median/p95 kinematic realization "
               f"error {kin_med:.3e}/{kin_p95:.3e}; pure covariance transport "
               f"error {peak_med:.3e}/{peak_p95:.3e} using entrant-native F on "
               f"{int(core.sum())} interior "
               f"particles after excluding {k:g} spacings from each face. "
               f"Worst health particle id {worst}; "
               "eigenvalue history plotted."],
        plots=[f"{scene['scene_id']}/{plot_name}"])
