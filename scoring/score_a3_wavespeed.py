"""A3 wave speed scorer (test_class: solution_verification).

Times the first motion of the far-end particle set after a prescribed
displacement pulse enters at the x-min face at t = 0, and compares
travel_distance / arrival_time against the thin-bar longitudinal speed
c = sqrt(E / rho).

Why the thin-bar formula and not the bulk P-wave speed: in a bar whose
lateral dimensions are much smaller than the pulse wavelength, the lateral
surfaces are traction-free and relax during passage, so the effective 1D
stiffness is Young's modulus E (uniaxial stress). The bulk speed
sqrt(E (1 - nu) / ((1 + nu)(1 - 2 nu) rho)) applies to laterally confined
media only. See Graff, "Wave Motion in Elastic Solids", chapter 2, or
Kolsky, "Stress Waves in Solids". The scene provenance repeats this.

Travel distance is measured from the driven face plane (rest x-min) to the
mean rest x of the far-end detection set, so the metric is self-consistent
with how arrival is detected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import common


def first_arrival_frame(disp_mag: np.ndarray, noise_floor: float) -> int:
    """Index of the first frame whose value exceeds the noise floor, or -1."""
    over = np.nonzero(disp_mag > noise_floor)[0]
    return int(over[0]) if len(over) else -1


def score(scene: dict, results_dir: Path, scenes_dir: Path) -> common.Verdict:
    traj = common.load_trajectory(results_dir)
    rest = common.load_scene_positions(scene, scenes_dir)
    common.check_initial_condition(traj, rest)
    common.check_output_fps(traj, scene)

    crit = scene["pass_criteria"]
    mat = scene["material"]

    x0 = rest[:, 0]
    x_lo, x_hi = x0.min(), x0.max()
    L_bar = x_hi - x_lo
    far = (x0 - x_lo) > crit["far_region_x_frac"] * L_bar
    if far.sum() == 0:
        raise common.SubmissionError("Far-end region selected zero particles.")

    disp = np.linalg.norm(traj.positions[:, far, :] - rest[None, far, :],
                          axis=2).mean(axis=1)
    k = first_arrival_frame(disp, crit["noise_floor_m"])
    if k <= 0:
        return common.verdict_from_scene(
            scene, common.STATUS_FAIL,
            measured={"arrival_detected": False},
            reference={"bar_wave_speed_m_s":
                       float(np.sqrt(mat["youngs_modulus_pa"]
                                     / mat["density_kg_m3"]))},
            tolerance={"rel_tolerance": crit["rel_tolerance"]},
            notes=["The far end never moved above the noise floor "
                   f"({crit['noise_floor_m']:g} m) within the run, or moved "
                   "at frame 0 (which would mean the initial condition was "
                   "not at rest). No wave arrived; the pulse boundary "
                   "condition may be missing."])

    t_arrival = float(traj.times[k])
    travel = float(x0[far].mean() - x_lo)
    c_meas = travel / t_arrival
    c_ref = float(np.sqrt(mat["youngs_modulus_pa"] / mat["density_kg_m3"]))
    rel_err = abs(c_meas - c_ref) / c_ref
    tol = crit["rel_tolerance"]

    return common.verdict_from_scene(
        scene,
        common.STATUS_PASS if rel_err < tol else common.STATUS_FAIL,
        measured={"bar_wave_speed_m_s": c_meas,
                  "arrival_time_s": t_arrival,
                  "travel_distance_m": travel,
                  "rel_error": float(rel_err)},
        reference={"bar_wave_speed_m_s": c_ref},
        tolerance={"rel_tolerance": tol},
        notes=[f"First motion of the far-end set (mean of {int(far.sum())} "
               f"particles) at frame {k}, t = {t_arrival:.4f} s."])
