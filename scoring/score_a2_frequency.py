"""A2 beam frequency scorer (test_class: solution_verification).

Measures the fundamental bending frequency of a plucked cantilever and
compares it to the Euler-Bernoulli analytical value computed from the scene's
material block (see reference/beams.py for the formula and citations).

Signal path: tip-region mean transverse displacement per frame, linear
detrend, Hann window, zero-padded FFT, dominant peak with parabolic
interpolation on log magnitude for sub-bin accuracy.

Regime guard: the Euler-Bernoulli reference assumes small deflections. If the
tip deflection ever exceeds the scene's declared fraction of beam length, the
verdict is INVALID (regime violated), not PASS or FAIL, because the reference
no longer applies to what was simulated.

Damping is reported as a diagnostic only. The scene specifies zero material
damping, so any measured decay is numerical dissipation of the submitting
system; publishing it alongside the frequency makes over-damped systems
visible without failing them here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import hilbert

from . import common
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from reference import beams


def dominant_frequency(signal: np.ndarray, fs: float) -> float:
    """Dominant frequency in Hz via Hann window + zero-padded FFT + parabolic
    peak interpolation. Expects a detrended signal."""
    n = len(signal)
    win = np.hanning(n)
    nfft = int(2 ** np.ceil(np.log2(n))) * 8   # 8x zero-pad for a fine grid
    spec = np.abs(np.fft.rfft(signal * win, n=nfft))
    k = int(np.argmax(spec[1:]) + 1)           # skip the DC bin
    # Parabolic interpolation on log magnitude around the peak bin.
    if 1 <= k < len(spec) - 1 and spec[k - 1] > 0 and spec[k + 1] > 0:
        la, lb, lc = np.log(spec[k - 1]), np.log(spec[k]), np.log(spec[k + 1])
        denom = la - 2.0 * lb + lc
        delta = 0.5 * (la - lc) / denom if denom != 0 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
    else:
        delta = 0.0
    return (k + delta) * fs / nfft


def damping_ratio(signal: np.ndarray, times: np.ndarray,
                  f_hz: float) -> float:
    """Report-only damping ratio from the Hilbert envelope.

    Fits log(envelope) linearly over the middle 80 percent of the run
    (the ends of the Hilbert transform suffer edge effects) and converts the
    decay rate lambda to a damping ratio zeta = lambda / (2*pi*f).
    """
    env = np.abs(hilbert(signal))
    lo, hi = int(0.1 * len(env)), int(0.9 * len(env))
    seg, t = env[lo:hi], times[lo:hi]
    if np.any(seg <= 0):
        return float("nan")
    slope = np.polyfit(t, np.log(seg), 1)[0]
    return float(-slope / (2.0 * np.pi * f_hz))


def score(scene: dict, results_dir: Path, scenes_dir: Path) -> common.Verdict:
    traj = common.load_trajectory(results_dir)
    rest = common.load_scene_positions(scene, scenes_dir)
    common.check_initial_condition(traj, rest)
    common.check_output_fps(traj, scene)

    crit = scene["pass_criteria"]
    L, w, h = scene["geometry"]["size_m"]
    mat = scene["material"]
    direction = np.asarray(scene["initial_conditions"]["direction"], float)

    # Tip region from REST configuration (scene positions, not the moving
    # frames): particles beyond the declared fraction of the length.
    x_min = rest[:, 0].min()
    tip = rest[:, 0] - x_min > crit["tip_region_rest_x_frac"] * L
    if tip.sum() == 0:
        raise common.SubmissionError("Tip region selected zero particles; "
                                     "scene geometry and positions disagree.")

    # Mean transverse displacement of the tip region per frame.
    disp = ((traj.positions[:, tip, :] - rest[None, tip, :]) @ direction
            ).mean(axis=1)

    # Regime guard BEFORE any frequency claim.
    max_defl = float(np.abs(disp).max())
    guard = crit["regime_max_tip_deflection_frac_of_length"] * L
    if max_defl > guard:
        return common.verdict_from_scene(
            scene, common.STATUS_INVALID,
            measured={"max_tip_deflection_m": max_defl},
            tolerance={"regime_max_tip_deflection_m": guard},
            notes=[
                f"Max tip deflection {max_defl:.4f} m exceeds the small-"
                f"deflection regime bound {guard:.4f} m. The Euler-Bernoulli "
                "reference does not apply, so no pass/fail judgment is made. "
                "Reduce the excitation response (check units and material "
                "mapping) and resubmit."])

    # Linear detrend, then measure.
    t = traj.times
    coeffs = np.polyfit(t, disp, 1)
    detrended = disp - np.polyval(coeffs, t)
    fs = 1.0 / float(np.median(np.diff(t)))
    f_meas = dominant_frequency(detrended, fs)

    f_ref = beams.cantilever_f1(mat["youngs_modulus_pa"],
                                mat["density_kg_m3"], L, w, h)
    rel_err = abs(f_meas - f_ref) / f_ref
    zeta = damping_ratio(detrended, t, f_meas)

    notes = [
        f"Damping ratio (report-only, numerical dissipation): zeta = {zeta:.4g}.",
        "Young's modulus used directly in the Euler-Bernoulli formula "
        "(uniaxial stress, lateral faces free; no E/(1-nu^2) correction).",
    ]
    tol = crit["rel_tolerance"]
    return common.verdict_from_scene(
        scene,
        common.STATUS_PASS if rel_err < tol else common.STATUS_FAIL,
        measured={"fundamental_frequency_hz": f_meas,
                  "rel_error": rel_err,
                  "damping_ratio": zeta,
                  "max_tip_deflection_m": max_defl},
        reference={"fundamental_frequency_hz": f_ref},
        tolerance={"rel_tolerance": tol},
        notes=notes)
