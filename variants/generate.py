"""Held-out scene variant generator.

    python variants/generate.py --scene a2_beam_frequency_v1 --seed 7

Produces a randomized variant of a public scene: material stiffness scaled
by a random factor in [0.5, 2], density in [0.8, 1.25], and geometry length
in [0.8, 1.2], with fresh frozen positions regenerated on the same lattice
spacing. The scene JSON's pass_criteria keeps the same reference NAME and
tolerances; the numeric reference is recomputed by the scorers from the
scaled material block at scoring time, exactly as for the public scene, so
nothing in the variant needs hand-tuned numbers.

The point: nobody can tune their simulator to the public scene instance.
Practice seeds may be committed; evaluation seeds are not. Generated files
land in variants/generated/, which is gitignored.

Supported scenes: box-geometry Tier A scenes (a2, a3, a5). Extending to a
new scene means teaching regenerate_positions() its geometry.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scenes"))

from generate_positions import lattice_box  # noqa: E402

SUPPORTED = ("a2_beam_frequency_v1", "a3_wavespeed_v1", "a5_stiffness_v1")

STIFFNESS_RANGE = (0.5, 2.0)
DENSITY_RANGE = (0.8, 1.25)
LENGTH_RANGE = (0.8, 1.2)


def regenerate_positions(scene: dict) -> np.ndarray:
    geom = scene["geometry"]
    if geom["type"] != "box":
        raise ValueError(f"Variant generation only supports box geometry, "
                         f"got '{geom['type']}'.")
    size = geom["size_m"]
    spacing = geom["particle_spacing_m"]
    # Keep the same anchoring convention as the base scenes: boxes start at
    # x = 0 and are centered on y and z.
    center = (size[0] / 2.0, 0.0, 0.0)
    # Exception: the a4-style floor-standing column would anchor differently,
    # but no such scene is in SUPPORTED.
    return lattice_box(size, spacing, center=center)


def generate(scene_id: str, seed: int, out_dir: Path) -> Path:
    base_path = REPO / "scenes" / f"{scene_id}.json"
    with open(base_path, "r", encoding="utf-8") as f:
        scene = json.load(f)
    if scene_id not in SUPPORTED:
        raise SystemExit(f"Scene {scene_id} does not support variants. "
                         f"Supported: {SUPPORTED}")

    rng = np.random.default_rng(seed)
    k_e = float(rng.uniform(*STIFFNESS_RANGE))
    k_rho = float(rng.uniform(*DENSITY_RANGE))
    k_len = float(rng.uniform(*LENGTH_RANGE))

    scene["material"]["youngs_modulus_pa"] *= k_e
    scene["material"]["density_kg_m3"] *= k_rho
    scene["geometry"]["size_m"][0] *= k_len

    variant_id = f"{scene_id}_seed{seed}"
    scene["scene_id"] = variant_id
    pos_name = f"{variant_id}_positions.npy"
    scene["geometry"]["particle_positions_file"] = pos_name
    scene.setdefault("provenance", {})["variant_of"] = scene_id
    scene["provenance"]["variant_scales"] = {
        "youngs_modulus": k_e, "density": k_rho, "length": k_len}

    # Scale any absolute x thresholds in boundary regions with the length so
    # clamp and pulse faces keep their meaning. Region strings in the base
    # scenes are of the form "x < 0.01" or "x > 0.095".
    for bc in scene.get("boundary_conditions", {}).values():
        if isinstance(bc, dict) and isinstance(bc.get("region"), str):
            op_parts = bc["region"].split()
            if len(op_parts) == 3 and op_parts[0] == "x":
                val = float(op_parts[2]) * k_len
                bc["region"] = f"x {op_parts[1]} {val:.6g}"

    out_dir.mkdir(parents=True, exist_ok=True)
    pts = regenerate_positions(scene)
    np.save(out_dir / pos_name, pts)
    out_path = out_dir / f"{variant_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scene, f, indent=2)
    print(f"variant: {out_path}")
    print(f"  E x {k_e:.3f}, rho x {k_rho:.3f}, length x {k_len:.3f}, "
          f"{len(pts)} particles")
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True, choices=SUPPORTED)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).parent / "generated")
    args = ap.parse_args(argv)
    generate(args.scene, args.seed, args.out_dir)


if __name__ == "__main__":
    main()
