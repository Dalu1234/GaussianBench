"""Create PhysGaussian's GaussianBench M1 material-validation submission.

PhysGaussian's ``jelly`` branch uses fixed-corotated elasticity:

    tau = 2 mu (F - R) F^T + lambda J (J - 1) I.

For homogeneous uniaxial deformation this can be evaluated directly without
introducing pull-rate, damping, or boundary-discretization artifacts.  The
transverse stretch is solved so that transverse stress is zero.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

REPO = Path(__file__).resolve().parent.parent
C10 = 708663.01
MU = 2.0 * C10
NU = 0.45
LAM = 2.0 * MU * NU / (1.0 - 2.0 * NU)


def transverse_pk1(transverse_stretch: float,
                   axial_stretch: float) -> float:
    jacobian = axial_stretch * transverse_stretch**2
    return (
        2.0 * MU * (transverse_stretch - 1.0)
        + LAM * (jacobian - 1.0) * jacobian / transverse_stretch
    )


def axial_cauchy(axial_stretch: float) -> float:
    transverse = brentq(
        transverse_pk1, 0.2, 1.0, args=(axial_stretch,))
    jacobian = axial_stretch * transverse**2
    axial_pk1 = (
        2.0 * MU * (axial_stretch - 1.0)
        + LAM * (jacobian - 1.0) * jacobian / axial_stretch
    )
    return axial_pk1 * axial_stretch / jacobian


def git_hash() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO,
        capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    out = args.out / "m1_material_validation_v1"
    out.mkdir(parents=True, exist_ok=True)

    stretches = np.concatenate(
        [np.arange(1.02, 1.5, 0.02), np.arange(1.5, 2.51, 0.05)])
    with open(out / "stress_strain.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stretch", "cauchy_stress_pa"])
        for stretch in stretches:
            writer.writerow(
                [f"{stretch:.4f}", f"{axial_cauchy(stretch):.1f}"])

    meta = {
        "system_name": "PhysGaussian",
        "system_version": git_hash(),
        "spec_version": "1.1.0",
        "hardware": platform.platform(),
        "notes": {
            "text": (
                "Direct homogeneous uniaxial evaluation of PhysGaussian's "
                "jelly fixed-corotated law as implemented in "
                "mpm_solver_warp/mpm_utils.py: tau = 2 mu (F-R) F^T + "
                "lambda J(J-1)I. mu=2*C10=1417326 Pa and nu=0.45; the "
                "transverse stretch is solved independently at every axial "
                "stretch so transverse PK1/Cauchy stress is zero. Direct "
                "constitutive evaluation is equivalent to the homogeneous "
                "quasi-static MPM limit and avoids pull-rate and fixture "
                "artifacts."
            )
        },
    }
    with open(out / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
