"""B1 analytical bimaterial bar in series.

The two regions have equal cross-sectional area and are loaded in series.
Under small-strain uniaxial stress, traction is continuous:

    sigma = Delta / (L1 / E1 + L2 / E2)
    epsilon_i = sigma / E_i.

The scorer estimates each regional axial strain from the slope of x(t)
against rest x over the central half of that material region.  These
symmetrical core windows exclude equal quarter-region bands next to the
clamps and interface.  It then checks both strain references and the
traction jump inferred from E_i * epsilon_i.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import common


def _fit_axial_strain(rest_x: np.ndarray, current_x: np.ndarray) -> float:
    x0 = rest_x - rest_x.mean()
    denom = float(np.dot(x0, x0))
    if denom <= 1e-20:
        raise common.SubmissionError("B1 regional strain window is degenerate.")
    slopes = ((current_x - current_x.mean(axis=1, keepdims=True)) @ x0) / denom
    return float(slopes.mean() - 1.0)


def score(scene: dict, results_dir: Path, scenes_dir: Path) -> common.Verdict:
    traj = common.load_trajectory(results_dir)
    rest = common.load_scene_positions(scene, scenes_dir)
    common.check_initial_condition(traj, rest)
    common.check_output_fps(traj, scene)

    mat = scene["material"]
    regions = mat["regions"]
    if len(regions) != 2:
        raise common.SubmissionError("B1 requires exactly two material regions.")
    E1 = float(regions[0]["youngs_modulus_pa"])
    E2 = float(regions[1]["youngs_modulus_pa"])
    xi = float(mat["interface_x_m"])

    x0 = rest[:, 0]
    x_lo, x_hi = float(x0.min()), float(x0.max())
    gauge = x_hi - x_lo
    spacing = float(scene["geometry"]["particle_spacing_m"])
    fixed = x0 - x_lo < 0.75 * spacing
    prescribed = x_hi - x0 < 0.75 * spacing

    crit = scene["pass_criteria"]
    n_hold = max(1, int(round(float(crit["hold_window_frac"])
                              * traj.n_frames)))
    hold = slice(traj.n_frames - n_hold, traj.n_frames)
    pos_hold = traj.positions[hold]

    ux_left = float((pos_hold[:, fixed, 0] - rest[None, fixed, 0]).mean())
    ux_right = float(
        (pos_hold[:, prescribed, 0] - rest[None, prescribed, 0]).mean())
    delta_meas = ux_right - ux_left
    delta_spec = float(
        scene["boundary_conditions"]["prescribed_face"]["displacement_m"][0])
    face_err = abs(delta_meas - delta_spec) / abs(delta_spec)
    if face_err > float(crit["prescribed_face_tolerance_rel"]):
        return common.verdict_from_scene(
            scene, common.STATUS_INVALID,
            measured={"face_displacement_m": delta_meas,
                      "face_displacement_rel_error": face_err},
            reference={"face_displacement_m": delta_spec},
            tolerance={"rel": crit["prescribed_face_tolerance_rel"]},
            notes=["The prescribed end displacement was not honored, so "
                   "the analytical bimaterial comparison is invalid."])

    L1 = xi - float(x0[fixed].mean())
    L2 = float(x0[prescribed].mean()) - xi
    sigma_ref = delta_meas / (L1 / E1 + L2 / E2)
    eps_ref = np.array([sigma_ref / E1, sigma_ref / E2])

    def core_mask(frac_pair):
        lo, hi = map(float, frac_pair)
        return ((x0 > x_lo + lo * gauge) & (x0 < x_lo + hi * gauge))

    left = core_mask(crit["left_core_x_frac"])
    right = core_mask(crit["right_core_x_frac"])
    eps = np.array([
        _fit_axial_strain(x0[left], pos_hold[:, left, 0]),
        _fit_axial_strain(x0[right], pos_hold[:, right, 0]),
    ])
    strain_rel = np.abs(eps - eps_ref) / np.maximum(np.abs(eps_ref), 1e-12)

    tractions = np.array([E1 * eps[0], E2 * eps[1]])
    traction_jump = (abs(tractions[0] - tractions[1])
                     / max(abs(tractions.mean()), 1e-12))

    strain_tol = float(crit["regional_strain_rel_tolerance"])
    traction_tol = float(crit["interface_traction_jump_rel_tolerance"])
    passed = bool(np.all(strain_rel < strain_tol)
                  and traction_jump < traction_tol)

    return common.verdict_from_scene(
        scene, common.STATUS_PASS if passed else common.STATUS_FAIL,
        measured={
            "left_strain": float(eps[0]),
            "right_strain": float(eps[1]),
            "left_strain_rel_error": float(strain_rel[0]),
            "right_strain_rel_error": float(strain_rel[1]),
            "left_inferred_traction_pa": float(tractions[0]),
            "right_inferred_traction_pa": float(tractions[1]),
            "interface_traction_jump_rel": float(traction_jump),
            "face_displacement_m": float(delta_meas),
        },
        reference={
            "left_strain": float(eps_ref[0]),
            "right_strain": float(eps_ref[1]),
            "continuous_traction_pa": float(sigma_ref),
        },
        tolerance={
            "regional_strain_rel": strain_tol,
            "interface_traction_jump_rel": traction_tol,
        },
        notes=[
            f"Core-window strains use x/L in "
            f"{crit['left_core_x_frac']} and {crit['right_core_x_frac']}.",
            "The left and right halves use different constitutive IDs; "
            "passing requires both the analytical strain jump and shared-"
            "grid traction continuity.",
        ])
