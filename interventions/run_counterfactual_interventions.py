"""Run the thermal and material-painting intervention experiments.

The experiments hold geometry and boundary conditions fixed while changing
one latent physical attribute:

1. thermal boundary temperature (370 K versus 400 K), graded by the
   Neumann/Stefan melt-front prefactor ratio;
2. bar material painting (uniform, soft-left/stiff-right, and the reversed
   painting), graded by the analytical fraction of end displacement taken
   up by the left half.

Usage (from the GaussianBench root):
    python interventions/run_counterfactual_interventions.py
    python interventions/run_counterfactual_interventions.py --skip-existing
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
SCENES = HERE / "scenes"
GAUSSIANFLESH = Path(
    os.environ.get("GAUSSIANFLESH_ROOT", BENCH.parent / "GaussianFlesh")
).resolve()
RESULTS = BENCH / "results" / "GaussianFlesh_counterfactual_2026-07-24"


def _run(scene_path: Path, skip_existing: bool) -> None:
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    out = RESULTS / scene["scene_id"] / "trajectory.parquet"
    if skip_existing and out.exists():
        print(f"[intervention] reuse {scene['scene_id']}", flush=True)
        return
    print(f"[intervention] run {scene['scene_id']}", flush=True)
    subprocess.run(
        [sys.executable, "-m", "export.run_bench_scene",
         "--scene", str(scene_path), "--out", str(RESULTS)],
        cwd=GAUSSIANFLESH, check=True)


def _thermal_rows(skip_existing: bool) -> dict:
    sys.path.insert(0, str(BENCH))
    from scoring import common
    from scoring.score_a4_meltfront import score
    from scoring.score_a4_meltfront import fit_sqrt_law

    def layer_crossing_fit(scene: dict) -> dict:
        """Fit front motion from per-layer 50% phase-crossing times.

        The bottom two particle layers lie inside the one-grid-cell
        fixed-temperature source band and are excluded.  This avoids grading
        the repeated-height staircase produced by frame-wise max-front
        sampling when a colder intervention melts only a few layers.
        """
        traj = common.load_trajectory(RESULTS / scene["scene_id"])
        rest = common.load_scene_positions(scene, SCENES)
        rest_z = rest[:, 2]
        rows = []
        for z_value in np.unique(rest_z):
            layer = np.isclose(rest_z, z_value)
            phase_fraction = traj.phase[:, layer].mean(axis=1)
            crossed = np.flatnonzero(phase_fraction >= 0.5)
            if crossed.size:
                rows.append((traj.times[crossed[0]],
                             z_value - float(rest_z.min())))
        crossings = np.asarray(rows, dtype=np.float64)
        moving = crossings[2:]
        if len(moving) < 4:
            raise RuntimeError(
                f"{scene['scene_id']} melted too few layers for the "
                "intervention fit.")
        prefactor, intercept, r2 = fit_sqrt_law(
            moving[:, 0], moving[:, 1])
        return {
            "layer_crossing_prefactor_m_per_sqrt_s": prefactor,
            "layer_crossing_intercept_m": intercept,
            "layer_crossing_r2": r2,
            "moving_layers_fit": len(moving),
        }

    rows = []
    for hot_k in (370.0, 400.0):
        scene_path = SCENES / f"i1_thermal_T{int(hot_k)}_v1.json"
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        _run(scene_path, skip_existing)
        verdict = score(scene, RESULTS / scene["scene_id"], SCENES)
        rows.append({
            "hot_floor_temperature_k": hot_k,
            "a4_status": verdict.status,
            "predicted_prefactor_m_per_sqrt_s":
                verdict.reference["front_prefactor_m_per_sqrt_s"],
            "framewise_prefactor_m_per_sqrt_s":
                verdict.measured["front_prefactor_m_per_sqrt_s"],
            "framewise_prefactor_rel_error": verdict.measured["rel_error"],
            "framewise_r2_vs_sqrt_t": verdict.measured["r2_vs_sqrt_t"],
            **layer_crossing_fit(scene),
        })
    pred_ratio = (rows[0]["predicted_prefactor_m_per_sqrt_s"]
                  / rows[1]["predicted_prefactor_m_per_sqrt_s"])
    meas_ratio = (rows[0]["layer_crossing_prefactor_m_per_sqrt_s"]
                  / rows[1]["layer_crossing_prefactor_m_per_sqrt_s"])
    ratio_err = abs(meas_ratio - pred_ratio) / pred_ratio
    return {
        "intervention": "hot-floor temperature",
        "held_fixed": ["geometry", "material", "initial state",
                       "boundary type", "simulation settings"],
        "readout": ("50%-phase layer-crossing fit; bottom two source-band "
                    "layers excluded"),
        "rows": rows,
        "predicted_prefactor_ratio_T370_over_T400": pred_ratio,
        "measured_prefactor_ratio_T370_over_T400": meas_ratio,
        "ratio_rel_error": ratio_err,
        "status": "PASS" if (
            ratio_err < 0.15
            and all(row["layer_crossing_r2"] > 0.98 for row in rows)
        ) else "FAIL",
        "tolerance": {
            "prefactor_ratio_rel_error": 0.15,
            "layer_crossing_r2_min": 0.98,
        },
    }


def _fit_left_extension_fraction(scene: dict, scene_id: str) -> dict:
    sys.path.insert(0, str(BENCH))
    from scoring import common

    rest = common.load_scene_positions(scene, SCENES)
    traj = common.load_trajectory(RESULTS / scene_id)
    x0 = rest[:, 0]
    x_lo, x_hi = float(x0.min()), float(x0.max())
    gauge = x_hi - x_lo
    spacing = float(scene["geometry"]["particle_spacing_m"])
    fixed = x0 - x_lo < 0.75 * spacing
    prescribed = x_hi - x0 < 0.75 * spacing
    n_hold = max(1, int(round(
        float(scene["pass_criteria"]["hold_window_frac"]) * traj.n_frames)))
    current = traj.positions[-n_hold:, :, 0]

    def core_strain(frac_pair) -> float:
        lo, hi = map(float, frac_pair)
        mask = ((x0 > x_lo + lo * gauge)
                & (x0 < x_lo + hi * gauge))
        centered = x0[mask] - x0[mask].mean()
        slopes = ((current[:, mask]
                   - current[:, mask].mean(axis=1, keepdims=True))
                  @ centered) / float(centered @ centered)
        return float(slopes.mean() - 1.0)

    crit = scene["pass_criteria"]
    eps_left = core_strain(crit["left_core_x_frac"])
    eps_right = core_strain(crit["right_core_x_frac"])

    regions = scene["material"]["regions"]
    E1 = float(regions[0]["youngs_modulus_pa"])
    E2 = float(regions[1]["youngs_modulus_pa"])
    xi = float(scene["material"]["interface_x_m"])
    L1 = xi - float(x0[fixed].mean())
    L2 = float(x0[prescribed].mean()) - xi
    predicted_fraction = (L1 / E1) / (L1 / E1 + L2 / E2)
    measured_fraction = ((eps_left * L1)
                         / (eps_left * L1 + eps_right * L2))
    absolute_error = abs(measured_fraction - predicted_fraction)
    return {
        "status": "PASS" if absolute_error < 0.05 else "FAIL",
        "left_modulus_pa": E1,
        "right_modulus_pa": E2,
        "predicted_left_extension_fraction": predicted_fraction,
        "measured_left_extension_fraction": measured_fraction,
        "absolute_fraction_error": absolute_error,
        "tolerance": {"absolute_fraction_error": 0.05},
    }


def _material_rows(skip_existing: bool) -> dict:
    rows = []
    for name in (
        "uniform_soft",
        "soft_left_stiff_right",
        "stiff_left_soft_right",
    ):
        scene_path = SCENES / f"i2_material_{name}_v1.json"
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        _run(scene_path, skip_existing)
        rows.append({"painting": name,
                     **_fit_left_extension_fraction(
                         scene, scene["scene_id"])})
    return {
        "intervention": "per-particle material painting",
        "held_fixed": ["geometry", "initial state", "prescribed extension",
                       "simulation settings"],
        "quantity": ("left-half fraction of the axial extension, inferred "
                     "from the two B1 core-strain gauges"),
        "rows": rows,
    }


def main() -> int:
    global GAUSSIANFLESH, RESULTS
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Reuse a trajectory when its output already exists.")
    parser.add_argument(
        "--gaussianflesh-root", type=Path, default=GAUSSIANFLESH,
        help="GaussianFlesh checkout containing export/run_bench_scene.py.")
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS,
        help="Output directory for the five intervention trajectories and summary.")
    args = parser.parse_args()
    GAUSSIANFLESH = args.gaussianflesh_root.resolve()
    RESULTS = args.results_dir.resolve()
    if not (GAUSSIANFLESH / "export" / "run_bench_scene.py").is_file():
        raise FileNotFoundError(
            f"GaussianFlesh runner not found under {GAUSSIANFLESH}. "
            "Pass --gaussianflesh-root or set GAUSSIANFLESH_ROOT.")
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "system": "GaussianFlesh",
        "thermal": _thermal_rows(args.skip_existing),
        "material_painting": _material_rows(args.skip_existing),
    }
    out = RESULTS / "counterfactual_interventions.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"[intervention] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
