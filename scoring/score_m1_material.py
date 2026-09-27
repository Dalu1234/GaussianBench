"""m1 material validation scorer (test_class: model_validation).

The suite's first validation against MEASURED physical data. The entrant
submits its own uniaxial stress-strain response (stress_strain.csv or
.parquet: columns 'stretch' and 'cauchy_stress_pa', transverse
traction-free); the scorer compares it to the measured natural-rubber curve
(the ASTM D412 Yeoh fit in reference/yeoh_rubber.py).

Why uniaxial: a structural test (a clamped beam, say) is dominated by
boundary-condition modeling, so a sim-vs-experiment gap there cannot be
attributed to the material. Uniaxial tension isolates the constitutive law,
which is what a model validation must do to mean anything.

Metric: median (and p95) relative error of the submitted stress against the
measured curve, over stretches in (1, max_stretch]. PASS when the median is
within median_rel_tolerance. The tolerance budgets the experiment's own
scatter (~6 percent) plus a constitutive-model allowance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from . import common

sys.path.insert(0, str(Path(__file__).parent.parent))
from reference import yeoh_rubber


def _load_submission(results_dir: Path):
    csv = results_dir / "stress_strain.csv"
    pq = results_dir / "stress_strain.parquet"
    import pandas as pd
    if csv.exists():
        df = pd.read_csv(csv)
    elif pq.exists():
        df = pd.read_parquet(pq)
    else:
        raise common.SubmissionError(
            "No stress_strain.csv or stress_strain.parquet in the submission. "
            "This model_validation scene requires the entrant's uniaxial Cauchy "
            "stress vs stretch (columns 'stretch' and 'cauchy_stress_pa', with "
            "transverse faces traction-free). A system that cannot report "
            "stress should not submit this scene; it is then marked NA.")
    for c in ("stretch", "cauchy_stress_pa"):
        if c not in df.columns:
            raise common.SubmissionError(
                f"stress_strain is missing required column '{c}'. "
                "Required: 'stretch', 'cauchy_stress_pa'.")
    return df


def score(scene: dict, results_dir: Path, scenes_dir: Path) -> common.Verdict:
    df = _load_submission(Path(results_dir))
    crit = scene["pass_criteria"]

    lam = df["stretch"].to_numpy(dtype=float)
    sig = df["cauchy_stress_pa"].to_numpy(dtype=float)
    if not np.all(np.isfinite(lam)) or not np.all(np.isfinite(sig)):
        raise common.SubmissionError("stress_strain contains non-finite values.")

    mask = (lam > 1.001) & (lam <= crit["max_stretch"])
    if int(mask.sum()) < crit["min_points"]:
        raise common.SubmissionError(
            f"Only {int(mask.sum())} points fall in the stretch range "
            f"(1, {crit['max_stretch']}]; at least {crit['min_points']} are "
            "required for a meaningful comparison.")
    lam, sig = lam[mask], sig[mask]

    ref = yeoh_rubber.uniaxial_cauchy_stress(lam)
    rel = np.abs(sig - ref) / np.abs(ref)
    med = float(np.median(rel))
    p95 = float(np.percentile(rel, 95))
    tol = crit["median_rel_tolerance"]

    return common.verdict_from_scene(
        scene,
        common.STATUS_PASS if med < tol else common.STATUS_FAIL,
        measured={"median_rel_error": med,
                  "p95_rel_error": p95,
                  "max_stretch_scored": float(lam.max()),
                  "max_strain_scored": float(lam.max() - 1.0),
                  "n_points": int(len(lam))},
        reference={"source": "Azarniya & Rahimi 2022, ASTM D412 Yeoh fit"},
        tolerance={"median_rel_tolerance": tol,
                   "max_stretch": crit["max_stretch"]},
        notes=[
            "Uniaxial tension isolates the constitutive law (no clamp, no "
            "bending), so this gap is cleanly attributable to the material "
            "model.",
            f"Compared against the measured natural-rubber curve up to "
            f"{lam.max():.2f}x stretch ({(lam.max() - 1) * 100:.0f}% "
            "engineering strain)."])
