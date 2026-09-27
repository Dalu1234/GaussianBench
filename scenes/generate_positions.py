"""Generate the frozen initial particle position .npy files for every scene.

Run from the repo root:
    python scenes/generate_positions.py

Deterministic: lattice constructions plus a fixed-seed jitter where noted.
The .npy files produced here are committed; entrants start from literally
these positions (SPEC.md section 3). If you change anything here, you are
creating new scene versions, bump the scene_id suffixes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SEED = 20260602
OUT = Path(__file__).parent


def lattice_box(size, spacing, center=(0.0, 0.0, 0.0)):
    """Regular cubic lattice filling a box, cell-centered, shape (N, 3)."""
    counts = [max(1, int(round(s / spacing))) for s in size]
    axes = [(np.arange(c) + 0.5) * s / c - s / 2.0
            for c, s in zip(counts, size)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return g + np.asarray(center, dtype=np.float64)


def lattice_ball(radius, spacing, center=(0.0, 0.0, 0.0)):
    box = lattice_box([2 * radius] * 3, spacing)
    box = box[np.linalg.norm(box, axis=1) <= radius]
    return box + np.asarray(center, dtype=np.float64)


def a1_ball():
    # Elastic ball for the conservation test. Radius 0.05 m, spacing 0.01 m.
    pts = lattice_ball(0.05, 0.01)
    np.save(OUT / "a1_ball_positions.npy", pts)
    print(f"a1_ball_positions.npy: {len(pts)} particles")


def a2_beam():
    # Slender cantilever 0.4 x 0.05 x 0.05 m, spacing 0.005 m, clamped end at x=0.
    pts = lattice_box([0.4, 0.05, 0.05], 0.005, center=(0.2, 0.0, 0.0))
    np.save(OUT / "a2_beam_positions.npy", pts)
    print(f"a2_beam_positions.npy: {len(pts)} particles")


def a3_bar():
    # Long thin bar 1.0 x 0.02 x 0.02 m along x, spacing 0.005 m.
    pts = lattice_box([1.0, 0.02, 0.02], 0.005, center=(0.5, 0.0, 0.0))
    np.save(OUT / "a3_bar_positions.npy", pts)
    print(f"a3_bar_positions.npy: {len(pts)} particles")


def a4_column():
    # Solid column 0.05 x 0.05 x 0.2 m standing on the hot floor at z=0.
    pts = lattice_box([0.05, 0.05, 0.2], 0.005, center=(0.0, 0.0, 0.1))
    np.save(OUT / "a4_column_positions.npy", pts)
    print(f"a4_column_positions.npy: {len(pts)} particles")


def a5_block():
    # Unit-aspect block 0.1 x 0.1 x 0.1 m for the Poisson stiffness test.
    pts = lattice_box([0.1, 0.1, 0.1], 0.005, center=(0.05, 0.0, 0.0))
    np.save(OUT / "a5_block_positions.npy", pts)
    print(f"a5_block_positions.npy: {len(pts)} particles")


def b1_bimaterial_bar():
    # Two-material bar in series, 0.1 x 0.04 x 0.04 m.  The interface is
    # x=0.05 m; cell-centered sampling places no particle exactly on it.
    pts = lattice_box([0.1, 0.04, 0.04], 0.005, center=(0.05, 0.0, 0.0))
    np.save(OUT / "b1_bimaterial_bar_positions.npy", pts)
    print(f"b1_bimaterial_bar_positions.npy: {len(pts)} particles")


def c1_block():
    # Block for the covariance push-forward test, same as A5 geometry.
    pts = lattice_box([0.1, 0.1, 0.1], 0.005, center=(0.0, 0.0, 0.0))
    np.save(OUT / "c1_block_positions.npy", pts)
    print(f"c1_block_positions.npy: {len(pts)} particles")


def c2_block():
    # Shear block, coarser spacing to keep Tier C submissions small.
    pts = lattice_box([0.1, 0.1, 0.1], 0.01, center=(0.0, 0.05, 0.0))
    np.save(OUT / "c2_block_positions.npy", pts)
    print(f"c2_block_positions.npy: {len(pts)} particles")


def c3_blob():
    # Procedural textured blob standing in for a captured Gaussian asset.
    # A captured asset should replace this when one with a clear license is
    # available (see scene description). Jittered ellipsoid, fixed seed.
    rng = np.random.default_rng(SEED)
    pts = lattice_ball(0.06, 0.008)
    pts = pts * np.array([1.0, 0.8, 1.3])          # squash into an ellipsoid
    pts += rng.normal(0.0, 0.0008, pts.shape)      # capture-like jitter
    pts[:, 1] -= pts[:, 1].min()                    # rest on the floor at y=0
    np.save(OUT / "c3_blob_positions.npy", pts)
    print(f"c3_blob_positions.npy: {len(pts)} particles")


def c4_asset():
    # Elongated irregular asset: a bent, noisy rod imitating a captured twig.
    # Frequency reference is recomputed from its actual bounding dimensions.
    rng = np.random.default_rng(SEED + 1)
    pts = lattice_box([0.4, 0.04, 0.04], 0.005, center=(0.2, 0.0, 0.0))
    # gentle spanwise bow plus surface noise so the shape is not a clean box
    pts[:, 1] += 0.015 * np.sin(np.pi * pts[:, 0] / 0.4)
    pts += rng.normal(0.0, 0.0006, pts.shape)
    np.save(OUT / "c4_asset_positions.npy", pts)
    print(f"c4_asset_positions.npy: {len(pts)} particles")


if __name__ == "__main__":
    a1_ball()
    a2_beam()
    a3_bar()
    a4_column()
    a5_block()
    b1_bimaterial_bar()
    c1_block()
    c2_block()
    c3_blob()
    c4_asset()
