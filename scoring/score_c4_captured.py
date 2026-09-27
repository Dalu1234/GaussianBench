"""C4 physics on captured geometry (test_class: capability).

The A2 frequency measurement applied to an irregular captured-style asset.
Signal path and regime guard are shared with score_a2_frequency; what changes
is the reference: the asset has no clean rectangular section, so the section
properties are measured from the particle distribution itself.

For any prismatic section, I = integral(y^2 dA) and A = integral(dA), hence
E I / (rho A) = E <y^2> / rho where <y^2> is the area-mean squared distance
from the neutral axis. For the bowed asset, <y^2> is computed per x-bin about
the local centroid (the neutral axis follows the bow), then averaged over
bins weighted by particle count. The scene provenance documents this as the
slenderness and irregularity correction. Length L is the rest bounding
extent along x. Wider tolerance (20 percent) per the scene file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import common
from .score_a2_frequency import damping_ratio, dominant_frequency

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from reference import beams


def measured_section_ratio(rest: np.ndarray, n_bins: int,
                           bend_axis: int = 1) -> float:
    """<y^2>: mean squared offset from the local per-bin centroid, m^2."""
    x = rest[:, 0]
    edges = np.linspace(x.min(), x.max() + 1e-12, n_bins + 1)
    which = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    y = rest[:, bend_axis]
    total, count = 0.0, 0
    for b in range(n_bins):
        sel = which == b
        if sel.sum() < 4:
            continue
        dy = y[sel] - y[sel].mean()
        total += float((dy ** 2).sum())
        count += int(sel.sum())
    if count == 0:
        raise common.SubmissionError("Centerline binning found no usable bins.")
    return total / count


def reference_frequency(scene: dict, rest: np.ndarray) -> float:
    mat = scene["material"]
    crit = scene["pass_criteria"]
    L = float(rest[:, 0].max() - rest[:, 0].min())
    y2 = measured_section_ratio(rest, int(crit["centerline_bins"]))
    return float(beams.BETA1_L ** 2 / (2.0 * np.pi * L ** 2)
                 * np.sqrt(mat["youngs_modulus_pa"] * y2
                           / mat["density_kg_m3"]))


def score(scene: dict, results_dir: Path, scenes_dir: Path) -> common.Verdict:
    traj = common.load_trajectory(results_dir)
    rest = common.load_scene_positions(scene, scenes_dir)
    common.check_initial_condition(traj, rest)
    common.check_output_fps(traj, scene)

    crit = scene["pass_criteria"]
    direction = np.asarray(scene["initial_conditions"]["direction"], float)
    L = float(rest[:, 0].max() - rest[:, 0].min())

    tip = (rest[:, 0] - rest[:, 0].min()) > crit["tip_region_rest_x_frac"] * L
    if tip.sum() == 0:
        raise common.SubmissionError("Tip region selected zero particles.")
    disp = ((traj.positions[:, tip, :] - rest[None, tip, :]) @ direction
            ).mean(axis=1)

    max_defl = float(np.abs(disp).max())
    guard = crit["regime_max_tip_deflection_frac_of_length"] * L
    if max_defl > guard:
        return common.verdict_from_scene(
            scene, common.STATUS_INVALID,
            measured={"max_tip_deflection_m": max_defl},
            tolerance={"regime_max_tip_deflection_m": guard},
            notes=["Tip deflection left the small-deflection regime; the "
                   "beam reference does not apply. No pass/fail judgment."])

    t = traj.times
    detrended = disp - np.polyval(np.polyfit(t, disp, 1), t)
    fs = 1.0 / float(np.median(np.diff(t)))
    f_meas = dominant_frequency(detrended, fs)
    f_ref = reference_frequency(scene, rest)
    rel_err = abs(f_meas - f_ref) / f_ref
    zeta = damping_ratio(detrended, t, f_meas)
    tol = crit["rel_tolerance"]

    return common.verdict_from_scene(
        scene,
        common.STATUS_PASS if rel_err < tol else common.STATUS_FAIL,
        measured={"fundamental_frequency_hz": f_meas,
                  "rel_error": float(rel_err),
                  "damping_ratio": zeta},
        reference={"fundamental_frequency_hz": f_ref,
                   "section_I_over_A_m2":
                       measured_section_ratio(rest,
                                              int(crit["centerline_bins"]))},
        tolerance={"rel_tolerance": tol},
        notes=["Reference computed from measured section properties of the "
               "irregular asset (see scene provenance); tolerance is wider "
               "than A2 accordingly."])
