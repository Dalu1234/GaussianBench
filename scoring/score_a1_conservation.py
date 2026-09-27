"""Conservation scorer (test_class: code_verification).

Serves both conservation scenes introduced by the v0.2.0 A1 split:
  a1a_conservation_inplace_v1   pure rigid spin, zero drift; isolates the
                                transfer scheme, tight tolerances
  a1b_conservation_drifting_v1  spin plus 11.5 m of drift; tests transfer
                                AND domain handling together, looser
                                tolerances plus a required meta.json
                                declaration of the domain strategy
(and the deprecated a1_conservation_v1, whose scene file is preserved
under scenes/deprecated/ for historical reproducibility).

Nothing here is hardcoded per scene_id: every tolerance, the optional
energy-fail semantics, and the optional declaration requirement come from
the scene's pass_criteria block.

Drift is measured as vector drift, max_t ||p(t) - p(0)|| / scale, not as
drift of the magnitude |p|, because a momentum vector that rotates while
keeping its magnitude is still a conservation violation. The scale is
||p(0)|| when nonzero. When the scene prescribes exactly zero net momentum
(a1a), the scale is the body's total momentum content sum_i m_i |v_i(0)|,
which for a rigid spin is positive and characterizes how much momentum the
transfers shuffle each step; drift relative to it is the honest analogue.

Scene-driven options in pass_criteria:
  linear_momentum_rel_drift_native / _derived   tolerance tiers
  angular_momentum_rel_drift                    tolerance
  energy_growth_flag_rel                        threshold for the energy note
  energy_monotone_growth_fails (optional)       when true, monotone kinetic
        energy growth beyond the flag threshold FAILS the scene instead of
        only being noted (a1a: with the body in place, sustained growth can
        only come from the integration scheme)
  requires_meta_declaration (optional)          generic mechanism, see
        common.check_meta_declarations; missing or invalid declarations
        score INVALID, and satisfied ones are echoed into
        measured["declared_<key>"] so reports can group results

A note on transfer schemes: Affine Particle-In-Cell (APIC) grid transfers
are designed for exact angular momentum conservation (Jiang et al., "The
Affine Particle-In-Cell Method", SIGGRAPH 2015), so MPM/APIC systems should
sit near machine precision on the L metric in a1a. Dissipative transfers
(pure PIC, or PIC/FLIP blends) show characteristic exponential decay. The
scorer records the |L| trace so the decay curve is visible either way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import common


def compute_momenta(traj: common.Trajectory, masses: np.ndarray):
    """Return (p, L, ke) traces.

    p:  (F, 3) total linear momentum, sum m_i v_i
    L:  (F, 3) SPIN angular momentum, about the instantaneous center of
        mass and relative to the center-of-mass velocity:
        sum m_i (x_i - x_cm(t)) cross (v_i - v_cm(t))
    ke: (F,) kinetic energy, 0.5 sum m_i |v_i|^2

    Erratum (v0.2.1): the v0.2.0 scorer measured L about the INITIAL
    center of mass. For a drifting body that definition is contaminated by
    any linear momentum drift through the cross term
    M (x_cm(t) - x_cm0) cross v_cm, which grows with the v0 * t lever arm:
    on the a1b scene a linear drift merely AT its own 1e-3 tolerance can
    induce an apparent L drift of order 3 (300x the L tolerance), so the
    old L metric measured nothing but the p drift. Spin angular momentum
    is equally conserved for an isolated body, is exactly immune to that
    cross term, and is what the scene always meant to grade; the p drift
    is already graded separately by its own metric.
    """
    v = traj.velocities
    x = traj.positions
    m = masses[None, :, None]
    M = masses.sum()
    p = (m * v).sum(axis=1)
    x_cm = (m * x).sum(axis=1) / M          # (F, 3) instantaneous cm
    v_cm = p / M
    r = x - x_cm[:, None, :]
    L = (m * np.cross(r, v - v_cm[:, None, :])).sum(axis=1)
    ke = 0.5 * (masses[None, :] * (v ** 2).sum(axis=2)).sum(axis=1)
    return p, L, ke


def _rel_vector_drift(series: np.ndarray, fallback_scale: float = 0.0) -> float:
    """max_t ||s(t) - s(0)|| / scale for an (F, 3) series.

    scale is ||s(0)||, or fallback_scale when the initial vector is
    (numerically) zero, as in the in-place scene's linear momentum.
    """
    ref = float(np.linalg.norm(series[0]))
    scale = ref if ref > 1e-12 * max(fallback_scale, 1.0) else fallback_scale
    if scale <= 0.0:
        raise common.SubmissionError(
            "Cannot normalize momentum drift: both the initial momentum and "
            "the fallback momentum-content scale are zero. Frame 0 "
            "velocities are missing or all zero, which contradicts the "
            "scene's initial spin.")
    return float(np.linalg.norm(series - series[0], axis=1).max() / scale)


def score(scene: dict, results_dir: Path, scenes_dir: Path) -> common.Verdict:
    # Required declarations first: without them the scene's result cannot
    # be interpreted, so no physics is even examined.
    declared, problems = common.check_meta_declarations(scene, results_dir)
    if problems:
        return common.verdict_from_scene(
            scene, common.STATUS_INVALID, notes=problems)

    traj = common.load_trajectory(results_dir)
    scene_pts = common.load_scene_positions(scene, scenes_dir)
    common.check_initial_condition(traj, scene_pts)
    common.check_output_fps(traj, scene)

    notes = []
    traj = traj.ensure_velocities()
    if traj.velocities_derived:
        notes.append(
            "Velocities were derived from positions by central finite "
            "differences; the looser derived-velocity tolerance tier applies.")
    masses, assumed = common.infer_masses(traj, scene)
    if assumed:
        notes.append(
            "No mass column submitted; equal per-particle masses assumed from "
            "scene density and geometry volume.")

    p, L, ke = compute_momenta(traj, masses)
    momentum_content = float(
        (masses * np.linalg.norm(traj.velocities[0], axis=1)).sum())
    p_drift = _rel_vector_drift(p, fallback_scale=momentum_content)
    L_drift = _rel_vector_drift(L)

    crit = scene["pass_criteria"]
    p_tol = (crit["linear_momentum_rel_drift_derived"]
             if traj.velocities_derived
             else crit["linear_momentum_rel_drift_native"])
    L_tol = crit["angular_momentum_rel_drift"]

    # Energy: bounded oscillation is fine (elastic exchange between kinetic
    # and strain energy). Monotone growth beyond the flag threshold marks an
    # unstable integrator; whether that fails the scene or is only noted is
    # the scene file's call (energy_monotone_growth_fails).
    ke_rel = ke / ke[0] - 1.0
    energy_growth = float(ke_rel[-1])
    growth_flag = crit["energy_growth_flag_rel"]
    monotone_growing = bool(np.all(np.diff(ke) >= -1e-12 * ke[0])
                            and energy_growth > growth_flag)
    energy_fails = bool(crit.get("energy_monotone_growth_fails", False)
                        and monotone_growing)
    if monotone_growing:
        notes.append(
            f"Kinetic energy grew monotonically by {energy_growth:.2%} over "
            "the run, indicating an unstable or energy-injecting integrator"
            + (" (FAIL per this scene's energy_monotone_growth_fails)."
               if energy_fails else " (report-only flag for this scene)."))

    checkpoints = np.linspace(0, traj.n_frames - 1, 5).astype(int)
    L_mag = np.linalg.norm(L, axis=1)
    decay_curve = {f"t={traj.times[k]:.2f}s": float(L_mag[k] / L_mag[0])
                   for k in checkpoints}
    notes.append(f"Angular momentum magnitude trace |L(t)|/|L(0)|: {decay_curve}")

    measured = {
        "linear_momentum_rel_drift": p_drift,
        "angular_momentum_rel_drift": L_drift,
        "kinetic_energy_rel_change_final": energy_growth,
    }
    for key, value in declared.items():
        measured[f"declared_{key}"] = value

    ok = p_drift < p_tol and L_drift < L_tol and not energy_fails
    return common.verdict_from_scene(
        scene,
        common.STATUS_PASS if ok else common.STATUS_FAIL,
        measured=measured,
        reference={"linear_momentum_rel_drift": 0.0,
                   "angular_momentum_rel_drift": 0.0},
        tolerance={"linear_momentum_rel_drift": p_tol,
                   "angular_momentum_rel_drift": L_tol,
                   "velocities_derived": traj.velocities_derived},
        notes=notes)
