"""Geometry sampling and mesh helper utilities for GaussianFlesh.

This module intentionally avoids importing the simulator so lightweight tests can
exercise geometry behavior without initializing Taichi or CUDA.
"""

import logging

import numpy as np
import pyvista as pv

LOGGER = logging.getLogger(__name__)

def fibonacci_sphere(n, r=1.0):
    """
    Place n points evenly on the SURFACE of a sphere using the Fibonacci spiral.

    The golden ratio controls the angular step so consecutive points never
    land on the same longitude line, giving a very uniform distribution with
    no clustering at the poles.  Used to sample the ball's surface for the
    drift measurement.
    """
    golden = (1 + np.sqrt(5)) / 2      # ≈ 1.618  -  the golden ratio
    idx    = np.arange(n, dtype=float)
    # theta: polar angle from north pole to south pole
    theta  = np.arccos(1 - 2*(idx+0.5)/n)
    # phi: longitude stepped by golden ratio to avoid repeating lines
    phi    = 2*np.pi*idx/golden
    return r * np.column_stack([
        np.sin(theta)*np.cos(phi),  # x
        np.cos(theta),              # y
        np.sin(theta)*np.sin(phi),  # z
    ])


def sample_sphere_volume(n, r=1.0, seed=42):
    """
    Place n points uniformly INSIDE a sphere of radius r.

    Strategy: repeatedly fill a cube with random points and keep only the
    ones that land inside the sphere (rejection sampling).  The cube has
    volume (2r)^3, the sphere π*r^3*4/3, so ~52 % of cube points are kept.
    Generating 4x more than needed each batch means we almost always have
    enough in one or two rounds.
    """
    rng = np.random.default_rng(seed)
    pts   = []
    total = 0
    while total < n:
        # fill a bounding cube with candidates; ~52% land inside the sphere,
        # so n*4 candidates yield ~2n inside points → usually one batch suffices
        batch  = rng.uniform(-r, r, size=(n * 4, 3))
        inside = batch[np.linalg.norm(batch, axis=1) <= r]
        if len(inside):
            pts.append(inside)
            total += len(inside)
    return np.concatenate(pts)[:n]   # trim to exactly n points


def sample_cuboid_volume(n, half_extents, seed=42):
    """
    Place n points uniformly inside an axis-aligned box centred at the origin.

    half_extents: (3,) array  -  half-width along each axis (x, y, z), so the
    full box spans [-hx, hx] × [-hy, hy] × [-hz, hz].
    """
    rng = np.random.default_rng(seed)
    he  = np.asarray(half_extents, dtype=float).ravel()
    if he.size != 3:
        raise ValueError("half_extents must have length 3")
    return rng.uniform(-he, he, size=(n, 3))

def sample_mesh_volume(mesh, n, seed=42):
    """Rejection-sample n points uniformly inside a (approximately) watertight PyVista mesh.
    Uses VTK's select_enclosed_points for the inside test  -  works on most clean OBJ meshes."""
    bounds = np.array(mesh.bounds)       # [xmin, xmax, ymin, ymax, zmin, zmax]
    lo, hi = bounds[0::2], bounds[1::2]
    rng    = np.random.default_rng(seed)
    pts    = []
    total  = 0
    while total < n:
        batch  = rng.uniform(lo, hi, size=(max(n * 6, 2000), 3))
        cloud  = pv.PolyData(batch.astype(np.float32))
        sel    = cloud.select_enclosed_points(mesh, tolerance=0.001, check_surface=False)
        inside = batch[sel.point_data["SelectedPoints"].astype(bool)]
        if len(inside):
            pts.append(inside)
            total += len(inside)
    return np.concatenate(pts)[:n]


def _in_ellipsoid(pts, centre, radii):
    """Return boolean mask: True where pts are inside the ellipsoid at centre with semi-axes radii."""
    d = (pts - centre) / radii
    return (d * d).sum(axis=1) <= 1.0


def sample_duck_volume(n, seed=42):
    """
    Place n points uniformly inside a procedural rubber-duck shape.

    The duck is built from four overlapping ellipsoids in rejection sampling:
      Body   -  wide squat ellipsoid, centred at origin
      Head   -  sphere offset up and forward
      Bill   -  flat ellipsoid projecting forward from the head
      Tail   -  small bump at the rear

    Coordinate system: Y = up, Z = forward (duck faces +Z).
    The duck fits inside roughly a 2×2×2.5 bounding box centred at origin,
    so it's comparable in scale to BALL_RADIUS=1.
    """
    rng   = np.random.default_rng(seed)
    # bounding box of the whole duck
    lo    = np.array([-1.1, -0.75, -1.0])
    hi    = np.array([ 1.1,  1.55,  1.05])
    pts   = []
    while sum(len(p) for p in pts) < n:
        batch = rng.uniform(lo, hi, size=(n * 8, 3))
        body  = _in_ellipsoid(batch, np.array([0.0,  0.0,   0.0]),  np.array([1.0, 0.65, 0.85]))
        head  = _in_ellipsoid(batch, np.array([0.0,  0.95,  0.25]), np.array([0.45, 0.45, 0.45]))
        bill  = _in_ellipsoid(batch, np.array([0.0,  0.80,  0.72]), np.array([0.22, 0.12, 0.28]))
        tail  = _in_ellipsoid(batch, np.array([0.0,  0.15, -0.88]), np.array([0.28, 0.30, 0.22]))
        inside = batch[body | head | bill | tail]
        pts.append(inside)
    return np.concatenate(pts)[:n]


# ============================================================
# PhysGaussian-style grid-based particle filling (--grid-fill)
# ============================================================
# Voxelize the shape's bounding box on a regular grid, identify interior
# cells via an inside-shape predicate, and fill each interior cell with
# `ppc` random particles.  Returns per-particle V0 = (cell_vol)/ppc and
# isotropic rest covariance σ ≈ (V0·3/4π)^(1/3).  Result: every particle
# perfectly represents its share of the shape regardless of total N  - 
# 1 particle → fills the shape; 100 → split into 100 covering blobs.


def _sphere_interior_check(R: float):
    """Returns a predicate fn: (n,3) array → bool[n] mask of pts inside sphere R."""
    R2 = R * R
    def check(pts: np.ndarray) -> np.ndarray:
        return (pts * pts).sum(axis=1) <= R2
    return check


def _cuboid_interior_check(half_extents: np.ndarray):
    """Predicate for an axis-aligned cuboid centred at origin."""
    h = np.asarray(half_extents, dtype=np.float64)
    def check(pts: np.ndarray) -> np.ndarray:
        return np.all(np.abs(pts) <= h, axis=1)
    return check


def _mesh_interior_check(mesh):
    """Predicate using PyVista's select_enclosed_points (watertight meshes)."""
    def check(pts: np.ndarray) -> np.ndarray:
        cloud = pv.PolyData(pts.astype(np.float32))
        sel   = cloud.select_enclosed_points(mesh, tolerance=0.001, check_surface=False)
        return np.asarray(sel.point_data["SelectedPoints"]).astype(bool)
    return check


def _duck_interior_check():
    """Predicate for the procedural duck (union of 4 ellipsoids)."""
    def check(pts: np.ndarray) -> np.ndarray:
        body = _in_ellipsoid(pts, np.array([0.0,  0.0,   0.0]),  np.array([1.0, 0.65, 0.85]))
        head = _in_ellipsoid(pts, np.array([0.0,  0.95,  0.25]), np.array([0.45, 0.45, 0.45]))
        bill = _in_ellipsoid(pts, np.array([0.0,  0.80,  0.72]), np.array([0.22, 0.12, 0.28]))
        tail = _in_ellipsoid(pts, np.array([0.0,  0.15, -0.88]), np.array([0.28, 0.30, 0.22]))
        return body | head | bill | tail
    return check


def fill_shape_grid_based(
    interior_check, bbox_lo: np.ndarray, bbox_hi: np.ndarray,
    target_n: int, ppc: int = 8, seed: int = 42,
):
    """PhysGaussian-style filling: voxelize bbox on a grid, fill each interior cell.

    Picks `grid_dx` so the result is approximately `target_n` particles assuming
    `ppc` per cell.  Each interior cell gets `ppc` random samples; each particle
    receives V0 = grid_dx³/ppc and an isotropic rest covariance with σ equal to
    the effective ownership radius (V0·3/4π)^(1/3).

    Returns:
        positions:  (n_actual, 3) float32
        V0_array:   (n_actual,)   float32   -  per-particle rest volume
        rest_covs:  (n_actual,3,3) float32  -  per-particle isotropic cov
        grid_dx:    float  -  fill grid spacing
        n_actual:   int    -  particle count (≈ target_n, may differ slightly)
    """
    rng = np.random.default_rng(seed)
    bbox_lo = np.asarray(bbox_lo, dtype=np.float64)
    bbox_hi = np.asarray(bbox_hi, dtype=np.float64)
    bbox_vol = float(np.prod(bbox_hi - bbox_lo))

    # Monte-Carlo estimate of shape volume fraction
    n_mc = 50000
    mc_pts = rng.uniform(bbox_lo, bbox_hi, size=(n_mc, 3))
    inside_frac = float(interior_check(mc_pts).mean())
    shape_vol = bbox_vol * max(inside_frac, 1e-6)

    # Solve for grid_dx so we hit ~target_n particles at ppc per cell
    n_cells_target = max(1, target_n // max(1, ppc))
    cell_vol = shape_vol / n_cells_target
    grid_dx = float(cell_vol ** (1.0 / 3.0))

    # Voxelize bbox at chosen dx
    span = bbox_hi - bbox_lo
    nx = max(1, int(np.ceil(span[0] / grid_dx)))
    ny = max(1, int(np.ceil(span[1] / grid_dx)))
    nz = max(1, int(np.ceil(span[2] / grid_dx)))
    # Test each cell centre for interior; use a single batched check.
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    cell_centers = (
        bbox_lo[None, :]
        + (np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1) + 0.5) * grid_dx
    )
    in_mask = interior_check(cell_centers)
    interior_centers = cell_centers[in_mask]
    n_interior = interior_centers.shape[0]

    if n_interior == 0:
        # Degenerate: tiny shape vs bbox.  Drop a single particle at bbox centre.
        pos    = ((bbox_lo + bbox_hi) * 0.5)[None, :]
        v_per  = max(shape_vol, 1e-6)
        sigma  = (v_per * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)
        covs   = np.eye(3, dtype=np.float32)[None, :, :] * (sigma * sigma)
        return pos.astype(np.float32), np.array([v_per], dtype=np.float32), covs, grid_dx, 1

    # Fill each interior cell with `ppc` random particles
    n_actual = n_interior * ppc
    cell_offsets = rng.uniform(-grid_dx * 0.5, grid_dx * 0.5, size=(n_interior, ppc, 3))
    positions = (interior_centers[:, None, :] + cell_offsets).reshape(-1, 3)

    # Per-particle V0 = cell_vol / ppc (uniform within this fill).
    v_per = (grid_dx ** 3) / ppc
    V0_array = np.full(n_actual, v_per, dtype=np.float32)

    # Per-particle covariance: isotropic, σ = (V0·3/4π)^(1/3) so the 1-σ ellipsoid
    # has roughly the volume the particle "owns" inside its cell.
    sigma = float((v_per * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0))
    rest_covs = np.zeros((n_actual, 3, 3), dtype=np.float32)
    rest_covs[:, 0, 0] = sigma * sigma
    rest_covs[:, 1, 1] = sigma * sigma
    rest_covs[:, 2, 2] = sigma * sigma

    return positions.astype(np.float32), V0_array, rest_covs, grid_dx, n_actual

def load_centered_mesh(path, scale=1.0, label="mesh"):
    mesh = pv.read(str(path)).clean().triangulate()
    centre = np.array(mesh.center)
    mesh.translate(-centre, inplace=True)
    if scale != 1.0:
        mesh.scale(scale, inplace=True)
    LOGGER.info(
        "[%s] Loaded %s  verts=%s  bounds=%s  volume=%.4f",
        label,
        path,
        mesh.n_points,
        np.array(mesh.bounds).round(3),
        mesh.volume,
    )
    return mesh


def _mesh_shape_rest(mesh, n):
    pts = sample_mesh_volume(mesh, n)
    vol = float(mesh.volume) if mesh.volume > 0 else float(
        np.prod(np.array(mesh.bounds)[1::2] - np.array(mesh.bounds)[0::2])
    )
    ext = np.array(mesh.bounds)[1::2] - np.array(mesh.bounds)[0::2]
    return pts, vol / n, float(max(ext)) / 12.0, float(mesh.bounds[3]) + 4.0


def _pad_or_truncate(pos, V0_arr, covs_arr, target_n: int, rng_seed: int = 1):
    """Make particle count exactly target_n.  If we're over, truncate; if under,
    duplicate random particles (their V0/cov stay the same  -  they re-fill cells
    they already share).  Used to keep N_PARTICLES field size predictable."""
    n_have = pos.shape[0]
    if n_have == target_n:
        return pos, V0_arr, covs_arr
    if n_have > target_n:
        idx = np.arange(target_n)
        return pos[idx], V0_arr[idx], covs_arr[idx]
    # n_have < target_n → duplicate samples to reach target
    rng = np.random.default_rng(rng_seed)
    deficit = target_n - n_have
    pick = rng.integers(0, n_have, size=deficit)
    pos2  = np.concatenate([pos, pos[pick]], axis=0)
    V02   = np.concatenate([V0_arr, V0_arr[pick]], axis=0)
    cov2  = np.concatenate([covs_arr, covs_arr[pick]], axis=0)
    return pos2, V02, cov2
