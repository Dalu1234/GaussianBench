"""A5 stiffness under displacement control (test_class: solution_verification).

WARNING to readers and submitters: this scene is uniaxial STRESS. The lateral
faces MUST be traction-free. If a submitting system clamps lateral boundaries
(for example a background grid boundary condition it cannot switch off), the
block cannot contract laterally, the Poisson check below reads near zero
contraction, and any force-based stiffness the system reports elsewhere will
read too stiff, because suppressing lateral contraction turns the measured
modulus into the oedometric (constrained) modulus
    M = E (1 - nu) / ((1 + nu)(1 - 2 nu))
which for nu = 0.3 is about 1.35x E. Submitters must declare their boundary
handling in meta.json notes.

Positions alone cannot give a reaction force, so this scorer measures the
apparent constitutive response kinematically: prescribe axial strain, then at
hold verify the free lateral faces contracted by Hooke's law for uniaxial
stress, lateral_strain = -nu * axial_strain. The force-based variant is
deliberately deferred to a future Tier B (see the scene description).

Method details:
  - Axial strain is measured from the actual mean displacement of the
    prescribed-face particles over the rest gauge length, not from the scene
    value, and the scorer first verifies the prescribed displacement was
    honored (else INVALID: the test did not run as specified).
  - Lateral strain is measured on a core region away from both loaded faces
    (Saint-Venant: end effects decay within about one cross-section width).
  - Uniform-scaling estimator: under homogeneous strain y -> (1 + eps) y,
    the mean absolute lateral coordinate scales by exactly (1 + eps), so
    eps_lat = mean|y|_hold / mean|y|_rest - 1, averaged over y and z and over
    the hold window to smooth residual ringing.
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
    bc = scene["boundary_conditions"]
    nu = float(scene["material"]["poisson_ratio"])
    delta_spec = float(bc["prescribed_face"]["displacement_m"][0])

    x0 = rest[:, 0]
    x_lo, x_hi = x0.min(), x0.max()
    gauge = x_hi - x_lo
    Lx = float(scene["geometry"]["size_m"][0])
    spacing = float(scene["geometry"]["particle_spacing_m"])

    fixed = x0 - x_lo < spacing * 0.75
    prescribed = x_hi - x0 < spacing * 0.75

    # Hold window: the last fraction of frames.
    n_hold = max(1, int(round(crit["hold_window_frac"] * traj.n_frames)))
    hold = slice(traj.n_frames - n_hold, traj.n_frames)

    # 1) Was the prescribed displacement honored?
    ux_face = (traj.positions[hold][:, prescribed, 0]
               - rest[None, prescribed, 0]).mean()
    face_err = abs(ux_face - delta_spec) / abs(delta_spec)
    if face_err > crit["prescribed_face_tolerance_rel"]:
        return common.verdict_from_scene(
            scene, common.STATUS_INVALID,
            measured={"prescribed_face_displacement_m": float(ux_face)},
            reference={"prescribed_face_displacement_m": delta_spec},
            tolerance={"rel": crit["prescribed_face_tolerance_rel"]},
            notes=["The prescribed face did not reach its commanded "
                   "displacement at hold, so the scene did not run as "
                   "specified and no pass/fail judgment is possible. Check "
                   "your displacement boundary condition implementation."])

    ux_fixed = (traj.positions[hold][:, fixed, 0]
                - rest[None, fixed, 0]).mean()
    eps_axial = float((ux_face - ux_fixed) / gauge)

    # 2) Lateral contraction on the core region (Saint-Venant guard).
    frac_lo, frac_hi = crit["core_region_x_frac"]
    core = ((x0 - x_lo > frac_lo * Lx) & (x0 - x_lo < frac_hi * Lx))
    eps_lats = []
    for axis in (1, 2):
        c0 = rest[core, axis]
        c0 = c0 - c0.mean()
        ref_extent = np.abs(c0).mean()
        ct = traj.positions[hold][:, core, axis]
        ct = ct - ct.mean(axis=1, keepdims=True)
        extent = np.abs(ct).mean()
        eps_lats.append(extent / ref_extent - 1.0)
    eps_lat = float(np.mean(eps_lats))

    eps_lat_ref = -nu * eps_axial
    rel_err = abs(eps_lat - eps_lat_ref) / abs(eps_lat_ref)
    tol = crit["rel_tolerance"]

    notes = [
        f"Axial strain at hold: {eps_axial:.5f} (from measured face "
        f"displacements over the rest gauge length {gauge:.4f} m).",
        f"Lateral strain measured on core region x in "
        f"[{frac_lo:.0%}, {frac_hi:.0%}] of length: eps_y={eps_lats[0]:.5f}, "
        f"eps_z={eps_lats[1]:.5f}.",
        "Reminder: lateral faces must be traction-free (uniaxial stress). "
        "Near-zero contraction usually means your lateral boundaries are "
        "clamped; declare boundary handling in meta.json.",
    ]
    return common.verdict_from_scene(
        scene,
        common.STATUS_PASS if rel_err < tol else common.STATUS_FAIL,
        measured={"lateral_strain": eps_lat, "axial_strain": eps_axial,
                  "rel_error": float(rel_err)},
        reference={"lateral_strain": eps_lat_ref, "poisson_ratio": nu},
        tolerance={"rel_tolerance": tol},
        notes=notes)
