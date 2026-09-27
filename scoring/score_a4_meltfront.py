"""A4 melt front scorer (test_class: capability, NA allowed).

Grades melt front propagation against the Neumann solution of the 1D Stefan
problem (reference/stefan.py, Carslaw and Jaeger chapter 11): the front
height must grow as s(t) = a * sqrt(t) with a = 2 lam sqrt(alpha).

Front measurement: per frame, the maximum z (height above the floor plane)
among particles with phase == 1. The phase column is REQUIRED. It cannot be
reconstructed from positions, because a molten and a solid particle at the
same location differ only in state; SPEC.md section 2.1 says the same.
Submitting this scene without a phase column is a format error; systems
without thermal support should not submit the scene at all and will be
marked NA by the runner.

Fit: least squares of front height against sqrt(t) with an intercept, after
discarding the initial transient (first transient_discard_frac of frames).
The intercept absorbs the half-layer bias of the particle-quantized front.
Pass requires both R^2 > r2_min (the growth law is actually diffusive) and
the fitted prefactor within prefactor_rel_tolerance of the Neumann value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from . import common

sys.path.insert(0, str(Path(__file__).parent.parent))
from reference import stefan


def fit_sqrt_law(times: np.ndarray, heights: np.ndarray):
    """Least-squares fit heights = a * sqrt(times) + b. Returns (a, b, r2)."""
    s = np.sqrt(times)
    A = np.stack([s, np.ones_like(s)], axis=1)
    coef, *_ = np.linalg.lstsq(A, heights, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    pred = A @ coef
    ss_res = float(((heights - pred) ** 2).sum())
    ss_tot = float(((heights - heights.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return a, b, r2


def score(scene: dict, results_dir: Path, scenes_dir: Path) -> common.Verdict:
    traj = common.load_trajectory(results_dir)
    rest = common.load_scene_positions(scene, scenes_dir)
    common.check_initial_condition(traj, rest)
    common.check_output_fps(traj, scene)

    if traj.phase is None:
        raise common.SubmissionError(
            "This scene requires the 'phase' column (0 solid, 1 fluid). The "
            "melt front cannot be inferred from positions alone. If your "
            "system has no phase state, do not submit this scene; it will "
            "be marked NA.")

    th = scene["material"]["thermal"]
    floor_z = float(rest[:, 2].min())

    # Front height per frame: max z among molten particles, 0 if none.
    molten = traj.phase == 1
    z = traj.positions[:, :, 2]
    heights = np.where(molten.any(axis=1),
                       np.where(molten, z, -np.inf).max(axis=1) - floor_z,
                       0.0)

    crit = scene["pass_criteria"]
    k0 = int(np.ceil(crit["transient_discard_frac"] * traj.n_frames))
    t_kept = traj.times[k0:]
    h_kept = heights[k0:]
    if not np.any(h_kept > 0):
        return common.verdict_from_scene(
            scene, common.STATUS_FAIL,
            measured={"melted_anything": False},
            notes=["No particle ever reached phase == 1 after the transient "
                   "window. Either the thermal coupling is inert or the "
                   "phase column is never set."])

    a_meas, b, r2 = fit_sqrt_law(t_kept, h_kept)
    a_ref = stefan.front_prefactor(
        th["conductivity_w_m_k"], scene["material"]["density_kg_m3"],
        th["specific_heat_j_kg_k"], th["latent_heat_j_kg"],
        scene["boundary_conditions"]["hot_floor"]["temperature_k"],
        th["melt_temperature_k"], th["initial_temperature_k"])
    rel_err = abs(a_meas - a_ref) / a_ref

    ok = r2 > crit["r2_min"] and rel_err < crit["prefactor_rel_tolerance"]
    return common.verdict_from_scene(
        scene,
        common.STATUS_PASS if ok else common.STATUS_FAIL,
        measured={"front_prefactor_m_per_sqrt_s": a_meas,
                  "fit_intercept_m": b,
                  "r2_vs_sqrt_t": r2,
                  "rel_error": float(rel_err),
                  "final_front_height_m": float(heights[-1])},
        reference={"front_prefactor_m_per_sqrt_s": float(a_ref)},
        tolerance={"r2_min": crit["r2_min"],
                   "prefactor_rel_tolerance": crit["prefactor_rel_tolerance"]},
        notes=[f"Fit window: frames {k0}..{traj.n_frames - 1} "
               f"(discarded the first {crit['transient_discard_frac']:.0%} "
               "as transient)."])
