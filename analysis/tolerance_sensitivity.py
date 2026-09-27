"""Re-score retained GaussianBench entries under scaled pass tolerances.

The scale multiplies an allowed error budget.  A scale below one is stricter;
a scale above one is more permissive.  Regime/validity gates (minimum travel,
minimum deformation, prescribed-face realization, frame count, and small-
deflection bounds) are intentionally frozen so FAIL/INVALID semantics do not
move during the analysis.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_all import discover_scenes, run_one  # noqa: E402


DEFAULT_ENTRIES = {
    "PhysGaussian": "results/PhysGaussian_native_2026-08-27",
    "GaussianFluent": "results/GaussianFluent_full",
    "OmniPhysGS": "results/OmniPhysGS_full",
    "PhysDreamer": "results/PhysDreamer_full",
    "Physics3D": "results/Physics3D_broad",
    "GASP": "results/GASP_broad",
}

# Maximum allowed errors or drifts.  These scale directly with the budget.
MAX_KEYS = {
    "linear_momentum_rel_drift_native",
    "linear_momentum_rel_drift_derived",
    "angular_momentum_rel_drift",
    "energy_growth_flag_rel",
    "rel_tolerance",
    "prefactor_rel_tolerance",
    "com_position_rel_error_max",
    "com_velocity_rel_error_max",
    "bbox_volume_rel_drift_max",
    "detF_nonpositive_fraction_max",
    "regional_strain_rel_tolerance",
    "interface_traction_jump_rel_tolerance",
    "kinematic_median_max",
    "kinematic_p95_max",
    "transport_median_max",
    "transport_p95_max",
    "peak_kinematic_median_rel_frobenius_max",
    "peak_kinematic_p95_rel_frobenius_max",
    "peak_transport_median_rel_frobenius_max",
    "peak_transport_p95_rel_frobenius_max",
    "condition_number_max",
    "degenerate_fraction_max",
    "edge_median_max",
    "edge_p95_max",
    "area_median_max",
    "area_p95_max",
    "path_straightness_max",
    "speed_variation_max",
    "tail_com_drift_max_rel",
    "transient_edge_p95_max",
    "tail_edge_median_max",
    "tail_area_median_max",
    "median_rel_tolerance",
}

# Quality scores with ideal value 1.  Scale their distance from the ideal.
UNIT_IDEAL_MIN_KEYS = {"r2_min", "ssim_min", "direction_cosine_min"}

# Collapse floors: a larger budget permits a proportionally smaller floor.
INVERSE_MIN_KEYS = {"min_eigenvalue_ratio"}

# Explicitly frozen regime gates.  Listed for auditability.
REGIME_KEYS = {
    "regime_max_tip_deflection_frac_of_length",
    "prescribed_face_tolerance_rel",
    "min_com_travel",
    "min_frames",
    "min_peak_area_p95",
    "squash_verify_height_frac_max",
}


def scaled_scene(scene: dict, scale: float) -> dict:
    out = copy.deepcopy(scene)
    criteria = out.get("pass_criteria", {})
    for key, value in list(criteria.items()):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if key in MAX_KEYS:
            criteria[key] = float(value) * scale
        elif key in UNIT_IDEAL_MIN_KEYS:
            criteria[key] = max(-1.0, min(1.0, 1.0 - scale * (1.0 - float(value))))
        elif key in INVERSE_MIN_KEYS:
            criteria[key] = float(value) / scale
        elif key == "psnr_min_db":
            # PSNR = -10 log10(MSE): multiplying allowed MSE by scale shifts
            # the threshold by -10 log10(scale).
            criteria[key] = float(value) - 10.0 * math.log10(scale)
    return out


def status_counts(rows: list[dict], scale: float, system: str) -> Counter:
    return Counter(r["status"] for r in rows
                   if r["scale"] == scale and r["system"] == system)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", nargs="+", type=float,
                        default=[0.5, 0.75, 1.0, 1.25, 2.0])
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results" / "tolerance_sensitivity")
    args = parser.parse_args(argv)
    if any(s <= 0 for s in args.scales):
        parser.error("all scales must be positive")

    scenes_dir = ROOT / "scenes"
    scenes = discover_scenes(scenes_dir)
    rows: list[dict] = []
    for system, rel_dir in DEFAULT_ENTRIES.items():
        result_dir = ROOT / rel_dir
        if not result_dir.exists():
            raise FileNotFoundError(result_dir)
        applicable = {p.name for p in result_dir.iterdir() if p.is_dir()}
        for scale in args.scales:
            for base_scene in scenes:
                if base_scene["scene_id"] not in applicable:
                    continue
                verdict = run_one(scaled_scene(base_scene, scale), result_dir,
                                  scenes_dir)
                rows.append({
                    "system": system,
                    "scene_id": verdict.scene_id,
                    "scale": scale,
                    "status": verdict.status,
                    "measured": verdict.measured,
                    "tolerance": verdict.tolerance,
                })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "interpretation": "scale multiplies allowed error; <1 stricter, >1 looser",
        "regime_gates_frozen": sorted(REGIME_KEYS),
        "entries": DEFAULT_ENTRIES,
        "scales": args.scales,
        "rows": rows,
    }
    (args.output_dir / "sensitivity.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    with (args.output_dir / "sensitivity.csv").open("w", newline="",
                                                     encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=(
            "system", "scene_id", "scale", "status"))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    baseline = 1.0
    if baseline not in args.scales:
        raise ValueError("the sweep must include scale 1.0")
    lines = [
        "# GaussianBench tolerance sensitivity",
        "",
        "The scale multiplies each verdict-bearing error budget. Values below "
        "one are stricter. Regime gates are fixed, so the sweep cannot turn a "
        "scene that was not realized into a valid result.",
        "",
        "| system | 0.5x P/F/I | 0.75x P/F/I | 1x P/F/I | 1.25x P/F/I | 2x P/F/I | flips vs 1x |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    baseline_map = {(r["system"], r["scene_id"]): r["status"]
                    for r in rows if r["scale"] == baseline}
    for system in DEFAULT_ENTRIES:
        cells = []
        flips = 0
        for scale in args.scales:
            counts = status_counts(rows, scale, system)
            cells.append(f"{counts['PASS']}/{counts['FAIL']}/{counts['INVALID']}")
            if scale != baseline:
                flips += sum(
                    r["status"] != baseline_map[(system, r["scene_id"])]
                    for r in rows if r["system"] == system and r["scale"] == scale)
        lines.append(f"| {system} | " + " | ".join(cells) + f" | {flips} |")

    lines += ["", "## Verdict transitions", ""]
    transitions = []
    for row in rows:
        base = baseline_map[(row["system"], row["scene_id"])]
        if row["scale"] != baseline and row["status"] != base:
            transitions.append(
                f"- {row['system']} / {row['scene_id']} at {row['scale']}x: "
                f"{base} -> {row['status']}")
    lines.extend(transitions or ["- None."])
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n",
                                                encoding="utf-8")
    print("\n".join(lines[:12]))
    print(f"\nWrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
