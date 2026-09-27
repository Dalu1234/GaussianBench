"""
gaussian_mesh_poc.py
Inverted MaGS: Gaussians -> mesh with continuum physics + per-particle deformation gradients.

Each frame (solids  -  TL-MPM):
  1. MPM grid forces (Total Lagrangian):
       F_p = WLS deformation gradient  (batch, no polar decomp)
       P_p = dPsi/dF  (full 3D first Piola-Kirchhoff stress)
       P2G: f_i = -V0 * sum_p P_p @ grad_w_ip   (scatter stress to grid)
       G2P: f_p = m_p * sum_i w_ip * f_i / m_i  (gather forces to particles)
     Grid is fixed in reference config; quadratic B-spline shape functions (27-node stencil).
     Swap material key to change Psi  -  particle struct, covariance, mesh binding unchanged.

Each frame (fluid  -  PBF, Gaussian Splashing pipeline):
  1. Predict positions under gravity.
  2. [×10] Compute density + λᵢ → position corrections → apply + floor clamp.
  3. Finalize velocity from Δx/dt; apply floor restitution.
  4. XSPH viscosity pass + linear damping.
  Particles are the rendering primitive  -  no mesh extraction, no covariance update.

Material models (full 3D Psi(F), swappable)
--------------------------------------------
  Corotated    R = UVᵀ from SVD(F),  P = 2mu*(F-R) + lam*(J-1)*J*F^{-T}   [default for solids]
  Neo-Hookean  P = mu*(F - F^{-T}) + lam*ln(J)*F^{-T}   J = det(F)        [legacy, kept for reference]
  StVK         E = (F^T F - I)/2,  S = mu*E + lam*tr(E)*I,  P = F @ S     [legacy, unstable at large strain]
  Fluid        Psi = kappa/2*(J-1)^2 + mu_reg/2*||F||^2,  P = kappa*(J-1)*J*F^{-T} + mu_reg*F

Controls
--------
  [Space]        drop ball
  [R]            reset
  [K]            kick
  [1/2/3/4/5]    Rubber / Metal / Jelly / Fluid / Wood
  [Up/Down]      mu_scale x1.2 / x0.8
  [S]            cycle shape: sphere → duck → cuboid → …
  [N]            toggle single-substep mode (1 MPM substep per display frame)
  [N] (paused)   advance one substep while paused
  [[ / ]]        slower / faster playback (wall-clock; physics dt unchanged)
  [0 / 9]        save / restore bookmark (camera + sim state)
  [F]            screenshot → captures/
  [V]            toggle MP4 recording → captures/
  [Q / Esc]      quit
"""

import sys as _sys_init
import json
# Same-module trick: when this file is run directly (`python gaussian_mesh_poc.py`)
# Python sets __name__ == "__main__" and the file's module name is "__main__".
# solver_tl / solver_ul then `import gaussian_mesh_poc as G`, which causes
# Python to LOAD THE FILE A SECOND TIME under the name "gaussian_mesh_poc".
# Result: ti.init() runs twice, all Taichi fields exist twice, and to_numpy()
# trips on a stale dtype.  Aliasing __main__ to "gaussian_mesh_poc" in
# sys.modules at the very start makes both names refer to the same instance.
if __name__ == "__main__":
    _sys_init.modules.setdefault("gaussian_mesh_poc", _sys_init.modules["__main__"])

import time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree             # fast nearest-neighbour lookups
from scipy.linalg import polar as scipy_polar  # polar decomposition F = R * U
from scipy.sparse import csr_matrix            # for component analysis on fracture
from scipy.sparse.csgraph import connected_components
from skimage.measure import marching_cubes     # extracts a surface mesh from a 3-D scalar field
import pyvista as pv                           # 3-D visualisation window
import taichi as ti                            # GPU-accelerated physics kernels
import pygame                                  # Xbox controller input
from gaussian_geometry import (
    _cuboid_interior_check,
    _duck_interior_check,
    _mesh_interior_check,
    _mesh_shape_rest,
    _pad_or_truncate,
    _sphere_interior_check,
    fill_shape_grid_based,
    fibonacci_sphere,
    load_centered_mesh,
    sample_cuboid_volume,
    sample_duck_volume,
    sample_mesh_volume,
    sample_sphere_volume,
)

# Initialise Taichi  -  uses CUDA on the RTX 4070.
ti.init(arch=ti.cuda, default_fp=ti.f32)

# ============================================================
# CLI  -  parsed early so field shapes can depend on args
# ============================================================
# Uses parse_known_args so importing this module (e.g. from collect_results.py)
# doesn't choke on the importer's own sys.argv.

import sys as _sys
from gaussian_cli import parse_cli

_CLI = parse_cli()

# ============================================================
# CONSTANTS
# ============================================================

# --- Solver / scene ---
SOLVER       = _CLI.solver
SCENE        = _CLI.scene
FICUS_DROP_COMPARE = bool(getattr(_CLI, "ficus_drop_compare", False))
FICUS_METAL_COMPARE = bool(getattr(_CLI, "ficus_metal_compare", False))

# Multi-material collide scene spec.
# Each entry: (material_key, shape, spawn_xyz, split_material_or_None)
# If split_material is set, the TOP HALF (rest-y > mid) of that entity is
# overwritten with the split material  -  giving a single body with TWO
# materials per-particle within it.  This is the "per-particle material
# identity within one connected object" demonstration.
_COLLIDE_ENTITIES = [
    # entity 0: metal cube, sitting on floor (impact target), uniform material
    ("2", "cuboid", ( 0.0, 1.05, 0.0), None),
    # entity 1: ONE sphere, bottom half rubber + top half jelly, drops onto metal.
    #           When it lands, the rubber bottom compresses moderately while the
    #           jelly top squashes dramatically  -  within a single connected body.
    ("1", "sphere", ( 0.0, 4.5,  0.0), "3"),
]

if FICUS_DROP_COMPARE or FICUS_METAL_COMPARE:
    N_ENTITIES     = 2
    ENTITY_SOLVERS = ["ul_mlsmpm", "ul_mlsmpm"]
    ENTITY_SHAPES  = None
    ENTITY_MATERIALS = None
elif SCENE == "collide":
    N_ENTITIES     = len(_COLLIDE_ENTITIES)
    ENTITY_SOLVERS = ["ul_mlsmpm"] * N_ENTITIES
    ENTITY_SHAPES  = [e[1] for e in _COLLIDE_ENTITIES]
    ENTITY_MATERIALS = [e[0] for e in _COLLIDE_ENTITIES]
elif SCENE == "compare":
    N_ENTITIES = 2
    ENTITY_SOLVERS = ["tl_apic", "ul_mlsmpm"]
    ENTITY_SHAPES  = None   # use SHAPE for both
    ENTITY_MATERIALS = None
else:
    N_ENTITIES = 1
    ENTITY_SOLVERS = [SOLVER]
    ENTITY_SHAPES  = None
    ENTITY_MATERIALS = None
N_PER_ENTITY = int(_CLI.n_per_entity)
FRAMES       = _CLI.frames if _CLI.frames is not None else (
    125 if (
        bool(getattr(_CLI, "physgaussian_ficus_match", False))
        or bool(getattr(_CLI, "physgaussian_ficus_drop", False))
    )
    else (3000 if SCENE in ("compare", "collide") else 800)
)
CSV_DIR      = _CLI.csv_dir
HEADLESS     = _CLI.headless
NO_SKIN      = _CLI.no_skin
HAND_ENABLED = _CLI.hand
GS_RENDER_ENABLED = bool(getattr(_CLI, "gs_render", False))
GS_ONLY           = bool(getattr(_CLI, "gs_only",   False))
VIDEO_OUT         = getattr(_CLI, "video", None)
PHYSGAUSSIAN_FICUS_DROP = bool(getattr(_CLI, "physgaussian_ficus_drop", False))
PHYSGAUSSIAN_FICUS_MATCH = (
    bool(getattr(_CLI, "physgaussian_ficus_match", False))
    or PHYSGAUSSIAN_FICUS_DROP
)
STATE_OUTPUT = getattr(_CLI, "state_output", None)
GRID_LIM_OVERRIDE = (2.0 if PHYSGAUSSIAN_FICUS_MATCH
                     else getattr(_CLI, "grid_lim", None))
GRID_V_DAMPING_SCALE = (0.9999 if PHYSGAUSSIAN_FICUS_MATCH
                        else float(getattr(_CLI, "grid_v_damping_scale", 1.0)))
PLY_UPRIGHT       = bool(getattr(_CLI, "ply_upright", False))
PLY_SPLIT         = getattr(_CLI, "ply_split", None)  # "MATKEY:FRACTION" or None
PLY_SPLIT_AXIS    = getattr(_CLI, "ply_split_axis", "y")
IMPACT_VY         = float(getattr(_CLI, "impact_vy", -4.0))
SQUASH_RELEASE    = bool(getattr(_CLI, "squash_release", False))
SQUASH_RATIO      = float(getattr(_CLI, "squash_ratio", 0.42))
SIDE_SQUASH_RELEASE = bool(getattr(_CLI, "side_squash_release", False))
DUAL_SQUASH_RELEASE = bool(getattr(_CLI, "dual_squash_release", False))
DRAG_DEMO        = bool(getattr(_CLI, "drag_demo", False))
GRAVITY_Y        = (0.0 if PHYSGAUSSIAN_FICUS_MATCH
                    else float(getattr(_CLI, "gravity_y", -9.8)))
if GS_ONLY or VIDEO_OUT:
    GS_RENDER_ENABLED = True   # implied
GRID_FILL_ENABLED = bool(getattr(_CLI, "grid_fill", False))
PPC = int(getattr(_CLI, "ppc", 8))
RECORD_PATH       = _CLI.record
CAPTURE_DIR       = Path(_CLI.capture_dir)
# ---- Thermodynamics (Stomakhin 2014) ----
THERMO_ENABLED    = bool(getattr(_CLI, "thermo", False))
AMBIENT_TEMP      = float(getattr(_CLI, "ambient_temp", 20.0))
HOT_FLOOR_TEMP    = getattr(_CLI, "hot_floor", None)   # None → material melt_temp + 60
HEAT_DIFFUSE_ITERS  = 12    # explicit Laplacian iterations per frame. 12 with conductivity=3
                            # gives a ~300-frame bottom-up melt front on a 2m cuboid.
                            # More iters = faster diffusion = gradient collapses in <50 frames.
HEAT_BAND_DX        = 0.5   # floor heat-source band thickness in grid cells. 0.5·dx keeps
                            # the Dirichlet zone to just the floor-contact layer so heat
                            # must diffuse upward, giving a visible bottom-up melt front.
DEMO_SPREAD_ENABLED = bool(getattr(_CLI, "demo_spread", False))
PRESSURE_PROJECT_ENABLED = bool(getattr(_CLI, "pressure_project", False))

# Webcam hand tracker (Tier 3: capsule-skeleton collider, 21 landmarks/hand x 2 hands).
# Image-space (MediaPipe normalised) -> world-space mapping for the hand bones.
# Defaults: un-mirror (user's right hand appears on world-right) + fixed z=0 plane.
HAND_X_HALF   = 2.5    # world x extent half-range (image x in [0,1] -> world x in [-HALF, +HALF])
HAND_Y_MIN    = 0.2    # world y at image-bottom
HAND_Y_MAX    = 5.0    # world y at image-top
# Depth (z) is derived from MediaPipe's relative per-landmark z.  MediaPipe z=0
# at the wrist; negative = closer to camera, positive = further away (in roughly
# image-x scale units).  We flip sign so reaching toward the camera maps to
# positive world-z (toward the viewer in our scene), and scale to span a useful
# z range.  Tune HAND_Z_SCALE to make depth movement more/less sensitive.
HAND_Z_CENTER = 0.0    # world-z where the wrist sits when MediaPipe z = 0
HAND_Z_SCALE  = -2.0   # world-z per unit MediaPipe z.  Was -6.0  -  too sensitive, the
                       # hand drifted ~1 m in z even when held still.  -2 keeps depth
                       # movement smaller than the ball radius so contact stays plausible.
HAND_K        = 12000.0 # capsule contact spring stiffness.  Was 4000  -  particles felt
                       # the bone but didn't get shoved hard.  12k ~= 2.5x the floor spring.
HAND_RADIUS   = 0.15   # bone capsule radius (metres)  -  was 6 cm (too thin; particles
                       # slipped between bones).  15 cm gives "boxing glove" fingers
                       # that catch every particle in the swept volume.
# MediaPipe loses track for 1-2 frames intermittently, causing the hand to pop
# in and out.  HAND_HOLD_FRAMES tells the viewport to keep the last good
# landmarks for this many frames after a loss before parking the hand off-screen.
# At TARGET_FPS=30 a hold of 8 gives ~0.27 s of grace.
HAND_HOLD_FRAMES = 8

# Fracture (continuum damage + WLS bond-break).  Enabled per-material via
# `yield_strain > 0` in MATERIALS, OR via --fracture for any material.
# Damage is permanent (ratchet only; never recovers).
FRACTURE_ENABLED       = _CLI.fracture
DAMAGE_YIELD_STRAIN    = 0.40    # strain at which damage starts growing
DAMAGE_RATE            = 5.0     # how fast damage accumulates per unit excess strain per second
BOND_BREAK_STRETCH     = 1.6     # WLS bond breaks at this stretch ratio (current/rest length)
BOND_BREAK_DMG_THRESH  = 0.7     # ... or when both endpoints exceed this damage level
# EMA smoothing factor on landmark positions when actively detected.
# 0.0 = no smoothing (raw, jittery).  1.0 = full hold (no movement).
# A value of 0.35 suppresses most MediaPipe jitter with about two frames of lag.
HAND_EMA      = 0.35

SHAPE          = _CLI.shape if _CLI.shape is not None else "sphere"   # "sphere" | "duck" | "cuboid" | "bunny" | "custom" | "ply"
PLY_PATH       = _CLI.ply

# When --ply is set, take particle count from the file (unless user explicitly
# capped it via --n-per-entity).  We peek the PLY header only  -  the full data
# load happens later in the geometry-setup block.
def _peek_ply_vertex_count(path: str):
    with open(path, "rb") as f:
        for _ in range(200):  # 3DGS PLY headers are < 100 lines
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header" or not line:
                break
    raise RuntimeError(f"could not find 'element vertex' in {path}")


def _peek_ply_opacity_count(path: str, threshold: float):
    with open(path, "rb") as f:
        properties = []
        vertex_count = 0
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property"):
                properties.append(line.split()[-1])
            elif line == "end_header":
                break
        opacity_idx = properties.index("opacity")
        arr = np.frombuffer(
            f.read(vertex_count * len(properties) * 4), dtype=np.float32
        ).reshape(vertex_count, len(properties))
    opacity = 1.0 / (1.0 + np.exp(-arr[:, opacity_idx]))
    return int(np.count_nonzero(opacity > threshold))

if PLY_PATH:
    SHAPE = "ply"
    _ply_file_count = _peek_ply_vertex_count(PLY_PATH)
    if PHYSGAUSSIAN_FICUS_MATCH:
        N_PER_ENTITY = _peek_ply_opacity_count(PLY_PATH, 0.02)
    # If the user didn't override --n-per-entity, cap at the CLI default (203930)
    # so we don't exceed the PLY count.  Pass --n-per-entity N to override.
    if "--n-per-entity" not in _sys.argv and not PHYSGAUSSIAN_FICUS_MATCH:
        N_PER_ENTITY = min(203930, _ply_file_count)
    print(f"[ply] {PLY_PATH}: {_ply_file_count} gaussians, using {N_PER_ENTITY}")

# N_PARTICLES is the TOTAL across all entities  -  preserved name so the existing
# field declarations and kernels still type-check.  Per-entity slicing happens
# explicitly via [start, end) ranges passed to the solver kernels.
N_PARTICLES    = N_PER_ENTITY * N_ENTITIES
BALL_RADIUS    = 1.0                            # radius of the ball in metres
# Cuboid rest shape: axis-aligned box [-hx,hx]×[-hy,hy]×[-hz,hz] centred at origin (metres).
CUBOID_HALF_EXTENTS = np.array([1.0, 1.0, 1.0])  # 2×2×2 m box  -  comparable to sphere diameter 2R
BUNNY_OBJ_PATH    = Path(__file__).resolve().parent / "assets" / "stanford-bunny.obj"
BUNNY_OBJ_SCALE   = 12.0
CUSTOM_OBJ_PATH  = Path(_CLI.custom_obj).expanduser() if _CLI.custom_obj else None
CUSTOM_OBJ_SCALE = 1.0   # uniform scale applied after auto-centering
BALL_START     = np.array([0.0, 4.0, 0.0])     # where the ball spawns (x, y, z)  -  4 m above ground
GRAVITY        = np.array([0.0, -9.8, 0.0])    # Earth gravity pulling downward (m/s²)
PARTICLE_MASS  = 1.0 / N_PARTICLES              # each particle carries an equal share of total mass 1 kg
DT             = ((0.008 if PHYSGAUSSIAN_FICUS_DROP else 0.04)
                  if PHYSGAUSSIAN_FICUS_MATCH else
                  (float(_CLI.frame_dt) if _CLI.frame_dt is not None else 1.0 / 60.0))
SUBSTEPS       = (max(1, int(round(DT / float(_CLI.substep_dt))))
                  if _CLI.substep_dt is not None else
                  ((80 if PHYSGAUSSIAN_FICUS_DROP else 400)
                   if PHYSGAUSSIAN_FICUS_MATCH else 40))
                                               # Combined stiffness check: (FLOOR_K + FLOOR_SPREAD) * dt_sub² / m_p
                                               # must stay well below 2.  At SUBSTEPS=20 combined ratio ≈ 1.09 (too close).
                                               # At SUBSTEPS=40 ratio ≈ 0.27  -  comfortable margin for all materials.
                                               # Also doubles elastic wave propagation per frame.
K_PHYS         = 8       # how many nearest neighbours each particle uses to measure its local deformation
MAX_SPEED      = 0.5 * (BALL_RADIUS / 4.0) / (DT / SUBSTEPS)   # CFL-based: no particle
                         # moves more than half a grid cell per substep.  Uses the
                         # default grid spacing (BALL_RADIUS/4); recomputed after
                         # shape-specific MPM_GRID_H is known (see below).
K_BIND         = _CLI.k_bind if _CLI.k_bind is not None else N_PER_ENTITY
                                # IDW skin: bind each mesh vertex to all entity particles by default.
                                # Override with --k-bind N for a cheaper k-nearest subset (e.g. 8 or 120).
TARGET_FPS     = 30      # rendering target  -  renders up to 30 times per second
DRIFT_GRID_RES = 32      # resolution of the density grid used to measure how much the mesh has drifted
DRIFT_ENABLED  = False   # expensive (marching cubes every 10 frames)  -  disable for demos

# V0 and MPM_GRID_H: sphere defaults  -  overridden below for duck / cuboid.
V0         = (4.0/3.0 * np.pi * BALL_RADIUS**3) / N_PARTICLES
MPM_GRID_H = BALL_RADIUS / 4.0

# Floor contact via penalty spring + one-way damper.
# FLOOR_K: spring stiffness per particle.  Must satisfy the explicit-Euler stability
#   condition: FLOOR_K * dt_sub² / m_p < 2  →  FLOOR_K < 2*(1/120)*(1200²) = 24000.
#   5000 gives ~6 substeps of contact, max ~18 mm penetration. Safe and gradual.
# FLOOR_FRIC: lateral drag while a particle is in contact with the floor.
# CONTACT_ZONE: height (metres) above the physics floor where contact force starts.
#   Any particle within this distance of the floor feels an upward force proportional
#   to its proximity.  This distributes the impact impulse across all particles in
#   the contact patch simultaneously  -  without waiting for the elastic wave to
#   propagate it from the 2–3 particles that actually cross y = 0.
#
#   Side-effect: at rest the ball "floats" with its bottom at y ≈ CONTACT_ZONE.
#   Compensated by raising the visual ground plane to y = CONTACT_ZONE so the
#   ball appears to sit on the floor.  (At rest, each bottom particle feels a
#   force FLOOR_K * ε → ball settles until total upward spring = total gravity,
#   which happens just inside the zone edge.)
FLOOR_K      = 5000.0
FLOOR_FRIC   = 5.0
CONTACT_ZONE = 0.05    # shallow contact zone  -  starts pushing at 5cm, minimal hover
# Settle damp: extra vertical decay applied only to slow particles in the contact zone.
# IMPORTANT: must be below the slowest bounce we want to preserve, otherwise it
# steals energy from active bounces as they pass through v=0.  At 0.5 m/s a ball
# can still bounce visibly ~5 times before the settle damp kicks in to stop
# residual floor oscillation.
# FLOOR_SETTLE_SPEED = 0 disables the per-particle sub-threshold velocity
# decay that was needed when soft materials creeped forever under viscous
# drag.  With proper material stiffness and zero damping, bodies settle via
# constitutive stress alone  -  no extra decay needed.
FLOOR_SETTLE_SPEED = 0.0
FLOOR_SETTLE_ALPHA = 0.0

# APIC_BLEND = 1.0 = pure APIC (Jiang 2015).  PhysGaussian uses pure APIC.
# A previous sub-1.0 blend compensated for C-mode energy accumulation.
# under soft materials  -  once materials are physically stiff, C-mode doesn't
# grow without bound and the blend is unnecessary.
APIC_BLEND = 1.0

# MPM_STRESS_CLAMP effectively disabled (1e12 = never triggers).  The clamp
# previously compensated for stress spikes when the SVD magnitude clamp [0.2, 5.0]
# fired  -  with the SVD clamp removed and materials properly scaled, stress
# stays in a normal range and the cap isn't needed.  Kept the field so the
# kernels don't need re-editing; just out of the normal-flow band.
MPM_STRESS_CLAMP = 1.0e12

FLOOR_SPREAD = 0.0

# ---- Material presets ----
# Each material is a dictionary of physical parameters.
#
# mu  (shear modulus)       -  resistance to shape change.  High mu = stiff (Metal).  Low = soft (Jelly).
# lam (first Lamé param)    -  resistance to volume change.  Usually set to ~5x mu.
# kappa                     -  bulk modulus for the fluid pressure law Ψ=κ/2·(J−1)².  0 for solid models.
# damping                   -  velocity drag coefficient.  terminal_vel = g*m_p / damping.
#                            Must be small so the ball isn't slowed before it hits the ground.
# restitution               -  fraction of vertical speed kept after bouncing (1 = perfect bounce, 0 = no bounce).
# cov_spread                -  how wide each Gaussian blob is in the rest pose (controls visual "puffiness").
# shape_alpha               -  how strongly the ball is pulled back toward its original sphere shape each frame.
#                            0 = no correction (jelly/fluid), >0 = spring back (rubber/metal).
# plastic_j2                -  if True, von Mises (J2) yield uses ‖dev P‖ instead of ‖P‖ (pressure-insensitive shear).
# tension_stretch_cap        -  max extra principal stretch (σ_max ≤ 1+cap); 0 = off. Caps brittle spike stretch.
# strain_rate_gamma         -  shear stiffening μ_eff = μ·(1 + γ‖C‖) using APIC velocity gradient C.
# mesh_color                -  display colour in the 3-D window.
#
# terminal_vel = g * m_p / damping = 9.8 / (120 * damping)
# Need terminal_vel >> 8.85 m/s (free-fall from 4 m) for proper bounce.
# Rule: damping << 0.0092.  Values below give terminal vel in parens.
# Material presets  -  physically scaled (PhysGaussian-aligned).
#
# (μ, λ) derived from Young's modulus E (Pa) and Poisson ratio ν:
#     μ = E / (2(1+ν))           shear modulus, Pa
#     λ = E·ν / ((1+ν)(1−2ν))    first Lamé, Pa
#
# density: kg/m³ (real-world units).  Particle mass derives as m_p = ρ · V_p.
# damping = 0 and shape_alpha = 0 across the board  -  PhysGaussian uses neither.
# Cohesion + rebound come entirely from the corotated constitutive law.
MATERIALS = {
    "1": {"name": "Rubber", "model": "corotated",
          # E=1e6, ν=0.45  (vulcanized rubber range)
          "mu": 3.45e5,  "lam": 3.10e6,  "kappa": 0,
          "damping": 0.0,    "restitution": 0.85,
          "cov_spread": 0.12, "shape_alpha": 0.0,
          "yield_stress": 1e9,  "relax_time": 0.0,
          "density": 1200.0,
          "friction": 0.80,
          "fracture": True,
          "yield_strain": 0.80,   # rubber is tough  -  high stretch before tearing
          "damage_rate": 4.0,
          "mesh_color": "#e63946"},
    "2": {"name": "Metal",  "model": "stvk",
          # PhysGaussian-style metal: StVK log-strain stress (Hencky) + log-strain
          # J2 return map + work hardening.  Mathematically textbook for finite
          # plasticity  -  return map overwrites F (Fp stays at I).  Hardening
          # makes the metal stiffer with each plastic event (xi controls rate).
          "mu": 7.14e7,  "lam": 1.07e8,  "kappa": 0,
          "damping": 0.0,    "restitution": 0.55,
          "cov_spread": 0.12, "shape_alpha": 0.0,
          "yield_stress": 5.0e7,    # ~50 MPa initial yield
          "relax_time": 0.0,
          "plastic_j2": 3,          # 3 = log-strain J2 with StVK (PhysGaussian metal)
          "hardening_xi": 10.0,     # work hardening rate (yield += 2μ·xi·Δγ on each plastic event)
          "density": 7800.0,
          "friction": 0.40,
          "fracture": True,
          "yield_strain": 1.20,
          "damage_rate": 3.0,
          "mesh_color": "#8ecae6"},
    "3": {"name": "Jelly",  "model": "corotated",
          # E=5e4, ν=0.45  -  real food-jelly stiffness (~50 kPa), 40× softer
          # than rubber.  Highly deformable, recovers via corotated stress.
          # Fracture enabled: damage ONLY grows when tear mode is active
          # (press T).  Normal drops/bounces are fully elastic  -  the
          # _ti_damage_update kernel checks _ti_tear_active before growing d.
          "mu": 1.72e4,  "lam": 1.55e5,  "kappa": 0,
          "damping": 0.5,    "restitution": 0.40,
          "cov_spread": 0.12, "shape_alpha": 0.0,
          "yield_stress": 1e9,  "relax_time": 0.0,
          "density": 1000.0,
          "friction": 0.90,
          "fracture": True,
          "yield_strain": 0.20,
          "damage_rate": 10.0,
          # Thermo: gelatin melts ~35°C.  Soft solid → flows when molten.
          "melt_temp": 35.0, "latent_heat": 30.0, "conductivity": 2.5,
          "heat_capacity": 1.0, "molten_kappa": 4.0e4, "softening_range": 10.0,
          "mesh_color": "#a8dadc"},
    "4": {"name": "Fluid",  "model": "fluid",
          "mu": 0.0,  "lam": 0,    "kappa": 0,
          "damping": 0.0,    "restitution": 0.05,
          "cov_spread": 0.10, "shape_alpha": 0.0,
          "yield_stress": 0.0,  "relax_time": 0.0,
          "density": 1000.0,
          "friction": 0.0,
          "mesh_color": "#457b9d"},
    "5": {"name": "Wood",   "model": "corotated",
          # E=1e7, ν=0.40
          "mu": 3.57e6,  "lam": 5.36e6,  "kappa": 0,
          "damping": 0.0,    "restitution": 0.45,
          "cov_spread": 0.12, "shape_alpha": 0.0,
          "yield_stress": 1e9,  "relax_time": 0.0,
          "density": 600.0,
          "friction": 0.50,
          "fracture": True,
          "yield_strain": 0.50,    # wood snaps at moderate stretch
          "damage_rate": 8.0,
          "mesh_color": "#8B5E3C"},
    "6": {"name": "Gas",    "model": "fluid",
          "mu": 0.0,  "lam": 0,    "kappa": 0.5,
          "damping": 0.0,   "restitution": 0.0,
          "cov_spread": 0.20, "shape_alpha": 0.0,
          "yield_stress": 0.0,  "relax_time": 0.0,
          "density": 1.225,
          "friction": 0.0,
          "mesh_color": "#aaffaa"},
    "c": {"name": "Clay",   "model": "sand",
          # Drucker-Prager constitutive model (friction-cone yield surface).
          # Neo-Hookean log-strain stress + DP return map in log-strain space.
          # Energy dissipation is INTRINSIC to the yield surface  -  no damping
          # coefficient needed.  Matches PhysGaussian sand material exactly.
          # friction_angle=30 -> alpha = sin(30)/(sqrt(3)*cos(30)) = 0.333
          "mu": 3.45e4,  "lam": 3.10e5,  "kappa": 0,
          "damping": 0.0,    "restitution": 0.05,
          "cov_spread": 0.12, "shape_alpha": 0.0,
          "yield_stress": 1e9,  "relax_time": 0.0,
          "plastic_j2": 2,             # 2 = Drucker-Prager mode in plasticity kernel
          "tension_stretch_cap": 0.0,
          "strain_rate_gamma": 0.0,
          "friction_angle": 30.0,      # internal friction angle (degrees)
          "density": 1800.0,
          "friction": 0.70,
          "fracture": True,
          "yield_strain": 0.30,    # clay tears easily
          "damage_rate": 8.0,
          "mesh_color": "#c77d4a"},
    "8": {"name": "Flesh",  "model": "corotated",
          # E=5e4, ν=0.40  (soft tissue, viscoelastic relax 1.2s)
          "mu": 1.79e4,  "lam": 7.14e4,  "kappa": 0,
          "damping": 0.0,    "restitution": 0.15,
          "cov_spread": 0.12, "shape_alpha": 0.0,
          "yield_stress": 1e9,  "relax_time": 1.2,
          "density": 1050.0,
          "friction": 0.85,
          "fracture": True,
          "yield_strain": 0.45,   # flesh tears at ~45% stretch
          "damage_rate": 7.0,
          "mesh_color": "#e8b4a0"},
    "b": {"name": "Bread",  "model": "corotated",
          # E=5e4, ν=0.40  -  soft elastic with continuum damage + WLS bond-break
          "mu": 1.79e4,  "lam": 7.14e4,  "kappa": 0,
          "damping": 0.0,    "restitution": 0.05,
          "cov_spread": 0.14, "shape_alpha": 0.0,
          "yield_stress": 1e9,  "relax_time": 0.0,
          "density": 400.0,
          "friction": 0.55,
          "fracture":     True,
          "yield_strain": 0.35,
          "damage_rate":  6.0,
          "mesh_color":   "#d4a574"},
    "w": {"name": "Wax",    "model": "corotated",
          # Soft waxy solid that melts cleanly into a flowing fluid.  The
          # thermodynamics demo material (Stomakhin 2014 melting-wax, Fig 5).
          # Melts ~60°C; molten phase is a compressible fluid (µ→0, κ bulk).
          "mu": 3.45e4,  "lam": 1.55e5,  "kappa": 0,
          "damping": 0.3,    "restitution": 0.05,
          "cov_spread": 0.12, "shape_alpha": 0.0,
          "yield_stress": 1e9,  "relax_time": 0.0,
          "density": 900.0,
          "friction": 0.05,   # near-frictionless on hot plate  -  lets melt spread laterally
          # Thermo params
          "melt_temp": 60.0, "latent_heat": 40.0, "conductivity": 3.0,
          "heat_capacity": 1.0, "molten_kappa": 4.0e5, "molten_mu": 3000.0,
          "softening_range": 20.0,
          "mesh_color":   "#e8d8a0"},
}

# Per-run physical overrides keep named presets intact for all other renders.
if _CLI.material in MATERIALS:
    _selected_mat = MATERIALS[_CLI.material]
    if _CLI.material_mu is not None:
        _selected_mat["mu"] = float(_CLI.material_mu)
    if _CLI.material_lambda is not None:
        _selected_mat["lam"] = float(_CLI.material_lambda)
    if _CLI.material_density is not None:
        _selected_mat["density"] = float(_CLI.material_density)
    if _CLI.material_damping is not None:
        _selected_mat["damping"] = float(_CLI.material_damping)

if PHYSGAUSSIAN_FICUS_MATCH:
    MATERIALS["1"].update({
        "name": "PhysGaussian Ficus FCR",
        "model": "corotated",
        "mu": 2.0e6 / (2.0 * (1.0 + 0.4)),
        "lam": 2.0e6 * 0.4 / ((1.0 + 0.4) * (1.0 - 2.0 * 0.4)),
        "density": 200.0,
        "damping": 0.0,
        "shape_alpha": 0.0,
        "friction": 0.0,
        "fracture": False,
    })

# ============================================================
# GEOMETRY
# ============================================================

def load_ply_3dgs(path: str, n: int, seed: int = 42):
    """Read a 3D Gaussian Splatting .ply file and return all trained attributes.

    The 3DGS PLY layout (62 float properties per vertex):
        x, y, z,                          # position
        nx, ny, nz,                       # normal (unused, typically zero)
        f_dc_0..f_dc_2,                   # DC term of spherical harmonics (base RGB)
        f_rest_0..f_rest_44,              # higher-order SH coefficients
        opacity,                          # logit-space opacity
        scale_0..scale_2,                 # log-space anisotropic scale
        rot_0..rot_3                      # rotation quaternion (w, x, y, z)

    Piece 2 (rendering): we additionally surface opacity (sigmoid-activated)
    and the full SH tensor in (N, 16, 3) layout for the diff_gaussian_rasterizer.

    The loaded scene is centered on the origin and rescaled so its widest axis
    fits in [-1, 1]  -  matches our usual simulation domain.  Σ is rescaled
    accordingly; opacity and SH are intensive properties (no rescale).

    Returns:
        positions:   (n, 3) float32, world-centered and rescaled
        covariances: (n, 3, 3) float32, rotated and rescaled to match positions
        opacities:   (n, 1) float32, sigmoid-activated [0, 1]
        shs:         (n, 16, 3) float32, DC + 15 higher-order coefficients per RGB channel
    """
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("unexpected EOF in PLY header")
            txt = line.decode("ascii", errors="ignore").strip()
            header_lines.append(txt)
            if txt == "end_header":
                break
        # Parse property names in order
        vertex_count = 0
        properties = []
        for txt in header_lines:
            if txt.startswith("element vertex"):
                vertex_count = int(txt.split()[-1])
            elif txt.startswith("property"):
                properties.append(txt.split()[-1])
        # 3DGS PLYs are float32 across the board
        bytes_per_vertex = len(properties) * 4
        raw = f.read(vertex_count * bytes_per_vertex)
        arr = np.frombuffer(raw, dtype=np.float32).reshape(vertex_count, len(properties))

    prop_idx = {name: i for i, name in enumerate(properties)}
    if PHYSGAUSSIAN_FICUS_MATCH:
        opacity_idx = prop_idx["opacity"]
        opacity_all = 1.0 / (1.0 + np.exp(-arr[:, opacity_idx]))
        arr = arr[opacity_all > 0.02]
        if len(arr) != n:
            raise RuntimeError(
                f"PhysGaussian opacity filtering produced {len(arr)} particles, expected {n}"
            )
    elif n < vertex_count:
        # Subsample to n particles (random, deterministic seed).
        sel = np.random.default_rng(seed).choice(vertex_count, size=n, replace=False)
        arr = arr[sel]

    positions = arr[:, [prop_idx["x"], prop_idx["y"], prop_idx["z"]]].astype(np.float32)
    # log-space scale → linear, then squared for the diag of Σ
    scale = np.exp(arr[:, [prop_idx["scale_0"], prop_idx["scale_1"], prop_idx["scale_2"]]])
    s2    = (scale ** 2).astype(np.float32)
    # quaternion (w, x, y, z), normalise to be safe
    q = arr[:, [prop_idx["rot_0"], prop_idx["rot_1"], prop_idx["rot_2"], prop_idx["rot_3"]]]
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = q.shape[0]
    R = np.empty((N, 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2*(y*y + z*z); R[:, 0, 1] = 2*(x*y - z*w); R[:, 0, 2] = 2*(x*z + y*w)
    R[:, 1, 0] = 2*(x*y + z*w);     R[:, 1, 1] = 1 - 2*(x*x + z*z); R[:, 1, 2] = 2*(y*z - x*w)
    R[:, 2, 0] = 2*(x*z - y*w);     R[:, 2, 1] = 2*(y*z + x*w);     R[:, 2, 2] = 1 - 2*(x*x + y*y)
    # Σ = R · diag(s²) · Rᵀ
    covs = np.einsum('nij,nj,nkj->nik', R, s2, R).astype(np.float32)

    # Opacity: PLY stores logit-space; rasterizer wants [0, 1].
    opac_logit = arr[:, prop_idx["opacity"]].astype(np.float32)
    opacities  = (1.0 / (1.0 + np.exp(-opac_logit))).reshape(-1, 1)

    # Spherical harmonics: DC term (3 coefficients, one per channel) +
    # 45 higher-order = 48 total, ordered as (N, 16, 3) for the rasterizer.
    dc = arr[:, [prop_idx["f_dc_0"], prop_idx["f_dc_1"], prop_idx["f_dc_2"]]].astype(np.float32)
    features_dc = dc.reshape(N, 1, 3)
    f_rest_names = [k for k in prop_idx if k.startswith("f_rest_")]
    f_rest_names.sort(key=lambda s: int(s.split("_")[-1]))
    f_rest_idx = [prop_idx[k] for k in f_rest_names]
    # PLY stores rest features channel-major: (N, 3, 15) → transpose to (N, 15, 3)
    features_rest = arr[:, f_rest_idx].astype(np.float32).reshape(N, 3, 15).transpose(0, 2, 1)
    shs = np.concatenate([features_dc, features_rest], axis=1).astype(np.float32)

    if PHYSGAUSSIAN_FICUS_MATCH:
        # Exact PhysGaussian transform2origin + shift2center111 preprocessing.
        lo = positions.min(axis=0)
        hi = positions.max(axis=0)
        scale_factor = 1.0 / float(np.max(hi - lo))
        positions = (positions - 0.5 * (lo + hi)) * scale_factor + 1.0
        covs *= scale_factor * scale_factor
        return positions, covs, opacities, shs

    # Center on origin
    positions -= positions.mean(axis=0)

    # Optional upright re-orientation (--ply-upright).  Trained 3DGS scenes
    # often use Z-up; our sim is Y-up with gravity along -Y.  Rotate the scene
    # so its trained-Z axis becomes world -Y (dense base at the bottom), giving
    # a natural pot-down drop.  R about X by +90°: (x,y,z) → (x, -z, y).
    # Covariances rotate as Σ' = R Σ Rᵀ.  SH left as-is (diffuse approximation;
    # view-dependent specular is negligible for foliage).
    if PLY_UPRIGHT:
        # R about X by -90°: (x,y,z) → (x, z, -y).  Puts trained-Z low (the pot)
        # at the bottom and trained-Z high (the leaves) at the top → pot-down.
        R = np.array([[1.0,  0.0, 0.0],
                      [0.0,  0.0, 1.0],
                      [0.0, -1.0, 0.0]], dtype=np.float32)
        positions = positions @ R.T
        covs = np.einsum("ij,njk,lk->nil", R, covs, R).astype(np.float32)

    # Rescale so the widest axis fits in [-1, 1]
    max_extent = float(np.abs(positions).max())
    if max_extent > 0:
        scale_factor = 1.0 / max_extent
        positions *= scale_factor
        covs      *= (scale_factor * scale_factor)

    return positions, covs, opacities, shs


def make_base_covs(positions, radial_spread=0.04, tangent_spread=0.18):
    """Build one rest covariance per particle, aligned to the radial direction."""
    covs = []
    for p in positions:
        n = p / np.linalg.norm(p)
        ref = np.array([0, 1, 0]) if abs(n[1]) < 0.9 else np.array([1, 0, 0])
        t1 = np.cross(n, ref)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        R = np.column_stack([n, t1, t2])
        D = np.diag([radial_spread, tangent_spread, tangent_spread])
        covs.append(R @ D @ R.T)
    return np.array(covs)


# ============================================================
# TOPOLOGY
# ============================================================

def _bspline2(s):
    """Quadratic B-spline weight over support [-1.5, 1.5]."""
    a = np.abs(s)
    return np.where(a < 0.5, 0.75 - a * a,
           np.where(a < 1.5, 0.5 * (1.5 - a) ** 2, 0.0))


def _dbspline2(s):
    """Derivative dw/ds of the quadratic B-spline above."""
    a = np.abs(s)
    return np.where(a < 0.5, -2.0 * s,
           np.where(a < 1.5, -(1.5 - a) * np.sign(s), 0.0))

# ============================================================
# MPM GRID  (Total Lagrangian, quadratic B-spline)
# ============================================================

def build_mpm_grid(rest_pos, grid_h=MPM_GRID_H):
    """
    Set up the invisible background grid used by the Material Point Method (MPM).

    What is MPM?
    ------------
    MPM is a hybrid particle/grid simulation method.  Particles carry the
    physical state (position, velocity, deformation).  The grid is a temporary
    scratch pad used to compute elastic forces in a way that naturally handles
    collisions and large deformations.

    Total Lagrangian means the grid is fixed in the REST (undeformed) configuration.
    The shape functions w_ip and their gradients dw_ip are computed once here and
    never change, which is both faster and more stable than the standard MPM
    approach of re-mapping the grid every frame.

    What we precompute per particle p for each of the 27 surrounding grid nodes i:
      node_idx  (N, 27)     -  which grid node index maps to position (ix, iy, iz)
      w_ip      (N, 27)     -  the weight: how much particle p "sees" node i
      dw_ip     (N, 27, 3)  -  the gradient of w_ip in the reference frame
      dpos_ip   (N, 27, 3)  -  world-space displacement from particle to each stencil node

    The 27-node stencil comes from a 3×3×3 cube of nodes around each particle
    using the B-spline support radius of 1.5 * grid_h.
    """
    # pad the grid slightly beyond the particles so boundary nodes exist
    pad      = 2.0 * grid_h
    gmin     = rest_pos.min(axis=0) - pad
    extent   = rest_pos.max(axis=0) - rest_pos.min(axis=0) + 2.0 * pad
    nx, ny, nz = (np.ceil(extent / grid_h)).astype(int) + 1
    n_grid   = int(nx * ny * nz)   # total number of grid nodes
    dims     = np.array([nx, ny, nz])

    # convert each particle's rest position to fractional grid coordinates
    p_gc  = (rest_pos - gmin) / grid_h       # (N, 3)  e.g. (2.4, 5.7, 1.1)
    # lower-left corner of the 3x3x3 stencil around this particle
    base  = np.floor(p_gc - 0.5).astype(int) # (N, 3)

    # offsets (0,0,0) through (2,2,2)  -  the 27 nodes of the stencil
    ox, oy, oz = np.meshgrid([0, 1, 2], [0, 1, 2], [0, 1, 2], indexing='ij')
    offsets    = np.stack([ox.ravel(), oy.ravel(), oz.ravel()], axis=1)  # (27, 3)

    N          = len(rest_pos)
    node_idx   = np.zeros((N, 27), dtype=np.int32)
    w_ip       = np.zeros((N, 27))
    dw_ip      = np.zeros((N, 27, 3))
    dpos_ip    = np.zeros((N, 27, 3))

    for k, o in enumerate(offsets):
        # grid-cell coordinates of this stencil node for every particle
        ijk  = base + o                                           # (N, 3)
        # mask out nodes that fall outside the grid (boundary particles)
        ok   = np.all((ijk >= 0) & (ijk < dims), axis=1)         # (N,)
        # convert (ix, iy, iz) to a flat 1-D index into the grid array
        lin  = ijk[:, 0] * ny * nz + ijk[:, 1] * nz + ijk[:, 2]
        node_idx[:, k] = np.where(ok, lin, 0)

        # s = signed distance from particle to this node in grid-cell units
        s    = p_gc - (base + o).astype(float)                   # (N, 3)

        # world-space displacement from particle to this stencil node
        dpos_ip[:, k, :] = -s * grid_h

        # 3-D weight = product of three 1-D B-spline weights (separable kernel)
        wx   = _bspline2(s[:, 0]); wy = _bspline2(s[:, 1]); wz = _bspline2(s[:, 2])
        w    = wx * wy * wz
        w_ip[:, k] = np.where(ok, w, 0.0)

        # gradient of the 3-D weight by the product rule:  d(wx*wy*wz)/dx = dwx*wy*wz
        # divide by grid_h to convert from "per grid cell" to "per metre"
        dwx  = _dbspline2(s[:, 0]); dwy = _dbspline2(s[:, 1]); dwz = _dbspline2(s[:, 2])
        dw_ip[:, k, 0] = np.where(ok, (dwx / grid_h) * wy * wz, 0.0)
        dw_ip[:, k, 1] = np.where(ok, wx * (dwy / grid_h) * wz, 0.0)
        dw_ip[:, k, 2] = np.where(ok, wx * wy * (dwz / grid_h), 0.0)

    return node_idx, w_ip, dw_ip, dpos_ip, n_grid


# ============================================================
# CONSTITUTIVE LAW  (full 3D first Piola-Kirchhoff stress)
# ============================================================

def compute_pk1_stress(Fs, mat, mu_scale=1.0):
    """
    Compute the first Piola-Kirchhoff stress P for every particle at once.

    P = dΨ/dF   -  the derivative of the stored elastic energy Ψ with respect to
    the deformation gradient F.  It tells us how much force the material generates
    when it is deformed.

    Inputs
    ------
    Fs       : (N, 3, 3) array  -  deformation gradient for each of the N particles.
               F maps vectors from rest space to current (deformed) space.
               F = I means no deformation (identity matrix).
    mat      : material parameter dict (mu, lam, model, …)
    mu_scale : runtime multiplier for mu (controlled by Up/Down keys)

    Returns
    -------
    Ps : (N, 3, 3)  -  PK1 stress for each particle.  Units: Pa (N/m²).

    ---- Why NOT StVK ----

    St. Venant-Kirchhoff (StVK) uses E = (FᵀF − I)/2 and S = μE + λ tr(E) I.
    The problem: E grows quadratically with stretch, so under the sudden large
    velocity discontinuity that impact creates (bottom particles bounce up while
    top particles are still falling), the stress can spike to values that
    exceed what the explicit integrator can handle in one substep.
    Worse, StVK's tangent stiffness (d²Ψ/dF²) can go NEGATIVE under large
    compression  -  the model actively pushes particles further apart instead of
    restoring them, causing the implosion you observed.

    ---- Neo-Hookean (used for ALL materials) ----

    Neo-Hookean has a provably positive-definite energy for any F with det(F) > 0.
    That means the tangent stiffness is always positive  -  the material ALWAYS
    resists deformation, no matter how extreme.  It is the standard model used
    in production MPM simulators (Disney snow, Houdini, etc.) for exactly this reason.

    Energy:  Ψ = μ/2 · (tr(FᵀF) − 3) − μ ln(J) + λ/2 · (ln J)²

    PK1 stress (derivative of Ψ with respect to F):
      P = μ (F − F⁻ᵀ) + λ ln(J) F⁻ᵀ

    where:
      J     = det(F)    -  volume ratio (J=1 at rest, J>1 expanded, J<1 compressed)
      F⁻ᵀ   = (F⁻¹)ᵀ   -  inverse-transpose, needed because ∂J/∂F = J · F⁻ᵀ
      ln(J)  -  logarithmic volume strain; zero at rest, sign tells compression/expansion

    Intuition for each term:
      μ (F − F⁻ᵀ):         resists shape change (shear); zero when F = I
      λ ln(J) F⁻ᵀ:         resists volume change; zero when J = 1 (no volume change)

    ---- Remaining vulnerability and how we close it ----

    The F⁻ᵀ term is still dangerous if a singular value of F goes very small.
    Even with det(F) clamped, a matrix can have det = 0.01 with one tiny
    singular value (e.g. 0.05), giving F⁻ᵀ entries of ~20 and a stress spike.
    The 1e-6 * I regularisation only helps if σ < 1e-6  -  it does nothing for
    σ = 0.05.

    Fix: SVD-clamp F before computing stress.  F = U Σ Vᵀ (singular value
    decomposition).  Clamping Σ to [σ_min, σ_max] gives a "sanitised" F̃ with
    bounded inverse and bounded stress, regardless of what the particles did.

      σ_min = 0.2  → maximum F⁻ᵀ entry ≤ 1/0.2 = 5   (hard compression limit)
      σ_max = 5.0  → maximum F entry ≤ 5               (hard stretch limit)

    This is exactly what the Disney MPM snow paper calls "clamp deformation
    gradient" and Houdini calls "max/min stretch."

    ---- Fluid branch ----

    Uses a weakly-compressible energy  Ψ = κ/2·(J−1)² + μ_reg/2·‖F‖²

      P_vol = κ(J−1)·J·F⁻ᵀ     (resists volume change, d²Ψ/dJ²=κ>0 everywhere)
      P_reg = μ_reg·F            (tiny shear regulariser  -  prevents rank deficiency)

    The old pressure law Ψ = κ(1−J+ln J) was concave for J>1 (d²Ψ/dJ²<0),
    so the material actively expanded under any perturbation.  The quadratic
    (J−1)² term is strictly convex for all J  -  stable by construction.
    """
    model = mat["model"]

    if model == "corotated":
        # Corotated linear elasticity (Genesis / Disney MPM style).
        #
        # Energy:  Ψ = μ||F − R||² + λ/2·(J−1)²
        # PK1:     P = 2μ(F − R) + λ(J−1)·J·F⁻ᵀ
        #
        # R = UVᵀ   -  the rotation part of F, extracted via SVD.
        # The term (F − R) measures deviation from pure rotation: zero stress
        # when the body is only rotating, restoring stress when it stretches
        # or compresses.  This gives implicit shape restoration without any
        # separate Procrustes correction step.
        #
        # Compared to Neo-Hookean:
        #   - More stable at moderate deformations (linear in stretch deviation)
        #   - Shape restoration is explicit in the energy rather than a post-step hack
        #   - Volumetric term (J−1)² is strictly convex (same as fluid model)
        mu_eff = mat["mu"] * mu_scale
        lam    = mat["lam"]

        U, s, Vh = np.linalg.svd(Fs)   # F = U diag(s) Vh

        # Stomakhin 2013 ("snow") improper-rotation fix.  np.linalg.svd returns
        # non-negative s, so when det(F) < 0 (inverted particle) the sign lives
        # in det(U @ Vh).  Flip last col of V (= last row of Vh) and last
        # singular value for inverted particles, so R = U @ Vh has det = +1
        # everywhere.  Clamp below renormalises the flipped sigma back to [0.2, 5.0].
        det_UVh = np.linalg.det(np.einsum('nij,njk->nik', U, Vh))
        mask    = det_UVh < 0
        if mask.any():
            s  = s.copy();  s[mask, 2]     = -s[mask, 2]
            Vh = Vh.copy(); Vh[mask, 2, :] = -Vh[mask, 2, :]

        # Rotation: R = U Vᵀ = U @ Vh  (now det=+1 everywhere)
        R = np.einsum('nij,njk->nik', U, Vh)          # (N, 3, 3)

        # SVD clamp for stable F⁻ᵀ  -  same limits as Neo-Hookean
        s_c  = np.clip(s, 0.2, 5.0)
        Fs_c = np.einsum('nij,nj,njk->nik', U, s_c, Vh)
        J    = s_c[:, 0] * s_c[:, 1] * s_c[:, 2]

        Finv_c  = np.linalg.inv(Fs_c)
        FinvT_c = Finv_c.transpose(0, 2, 1)

        # P = 2μ(F̃ − R) + λ(J−1)·J·F̃⁻ᵀ
        return (2.0 * mu_eff * (Fs_c - R)
                + lam * (J - 1.0)[:, None, None] * J[:, None, None] * FinvT_c)

    elif model == "neohook":
        mu_eff = mat["mu"] * mu_scale   # allow runtime stiffness tuning
        lam    = mat["lam"]

        # --- SVD clamp: F = U Σ Vᵀ, clamp singular values, reconstruct F̃ ---
        #
        # Singular values Σ = [σ₁, σ₂, σ₃] are the three principal stretch
        # ratios.  σ = 1 means no stretch on that axis.  σ = 0.05 means
        # compressed to 5% of rest length  -  pathological and causes F⁻ᵀ = 20×.
        #
        # np.linalg.svd returns (U, s, Vh) where s contains the singular values
        # and Vh is already the transpose of V (so F = U @ diag(s) @ Vh).
        U, s, Vh = np.linalg.svd(Fs)                  # U,Vh: (N,3,3)   s: (N,3)

        # Clamp: prevent any axis being compressed below 20% or stretched beyond 500%.
        # These are generous limits that stop blow-ups while allowing realistic
        # large deformations (a rubber ball flattens to ~40% height on impact).
        s_clamped = np.clip(s, 0.2, 5.0)              # (N, 3)

        # Reconstruct the sanitised F̃ = U diag(σ_clamped) Vᵀ
        # einsum 'nij,nj,njk->nik': for particle n, F̃ = U @ diag(s) @ Vh
        Fs_c = np.einsum('nij,nj,njk->nik', U, s_clamped, Vh)  # (N, 3, 3)

        # J = det(F̃) = product of clamped singular values.
        # Using the clamped F̃ means J is always in [0.2³, 5³] = [0.008, 125].
        J   = s_clamped[:, 0] * s_clamped[:, 1] * s_clamped[:, 2]  # (N,)

        # F̃⁻ᵀ: singular values of the inverse are 1/σ, bounded by [1/5, 1/0.2] = [0.2, 5]
        # because sigma was clamped. The former 1e-6 regularization is no longer required.
        Finv_c  = np.linalg.inv(Fs_c)                 # (N, 3, 3)   -  safe: min σ = 0.2
        FinvT_c = Finv_c.transpose(0, 2, 1)           # (N, 3, 3)

        # ln(J): logarithmic volume strain on clamped F̃.
        # Bounded: ln(0.2³) = −4.8  to  ln(5³) = +4.8.
        lnJ = np.log(J)                                # (N,)

        # P = μ(F̃ − F̃⁻ᵀ) + λ ln(J) F̃⁻ᵀ
        # All terms are now bounded by construction  -  no spike possible.
        return mu_eff * (Fs_c - FinvT_c) + lam * lnJ[:, None, None] * FinvT_c  # (N, 3, 3)

    elif model == "stvk":
        # ---- LEGACY  -  kept for reference / comparison ----
        # StVK blows up under large strain at impact because its tangent
        # stiffness goes negative in compression.  Use "neohook" instead.
        mu_eff = mat["mu"] * mu_scale
        lam    = mat["lam"]
        FtF    = np.einsum('nij,nik->njk', Fs, Fs)
        E      = 0.5 * (FtF - np.eye(3))
        trE    = np.trace(E, axis1=1, axis2=2)
        S      = mu_eff * E + lam * trE[:, None, None] * np.eye(3)
        return np.einsum('nij,njk->nik', Fs, S)

    elif model == "fluid":
        # Weakly-compressible fluid:  Ψ = κ/2·(J−1)² + μ_reg/2·‖F‖²
        # P = κ(J−1)·J·F⁻ᵀ  +  μ_reg·F
        # κ controls bulk stiffness (nearly incompressible).
        # μ_reg is a tiny shear regulariser that prevents F from becoming
        # rank-deficient as the fluid deforms freely  -  NOT a solid shear modulus.
        kappa  = mat["kappa"]
        mu_reg = mat["mu"] * mu_scale

        # SVD clamp  -  same protection as Neo-Hookean
        U, s, Vh = np.linalg.svd(Fs)
        s_c  = np.clip(s, 0.2, 5.0)
        Fs_c = np.einsum('nij,nj,njk->nik', U, s_c, Vh)
        J    = s_c[:, 0] * s_c[:, 1] * s_c[:, 2]

        Finv_c  = np.linalg.inv(Fs_c)
        FinvT_c = Finv_c.transpose(0, 2, 1)

        P_vol = kappa * (J - 1.0)[:, None, None] * J[:, None, None] * FinvT_c
        P_reg = mu_reg * Fs_c
        return P_vol + P_reg

    else:
        raise ValueError(f"Unknown model: {model!r}")


def compute_mpm_forces(Fs, node_idx, w_ip, dw_ip, n_grid, mat, mu_scale=1.0):
    """
    Compute elastic forces on every particle using the MPM grid transfer.

    Why use a grid at all?
    ----------------------
    If we computed forces directly between particle pairs (like a spring system),
    we'd need to handle every combination  -  O(N²) work, and getting the stresses
    right for a continuum material is tricky.  Instead, MPM uses a background grid
    as a shared medium:

      Step 1 (P2G  -  Particles to Grid):
        Each particle scatters its stress onto the 27 nearby grid nodes,
        weighted by the B-spline shape function gradients.

      Step 2 (Grid solve):
        Each grid node accumulates mass and force from all nearby particles,
        then computes an acceleration  a = f / m.

      Step 3 (G2P  -  Grid to Particles):
        Each particle gathers accelerations back from its 27 grid nodes,
        weighted by the B-spline weights.  This becomes the elastic force on
        the particle.

    The force formula comes from the principle of virtual work:
      f_i (grid) = -V0 · Σ_p  P_p @ ∇_X w_ip
      f_p (particle) = m_p · Σ_i  w_ip · a_i

    where:
      V0      = reference volume per particle
      P_p     = first Piola-Kirchhoff stress of particle p
      ∇_X w_ip = gradient of shape function (how fast weight changes with position)
    """
    # --- Step 1: Compute PK1 stress for every particle ---
    Ps = compute_pk1_stress(Fs, mat, mu_scale)   # (N, 3, 3)

    # Clamp per-particle stress magnitude before scatter.
    # Extreme stress (e.g. a heavily-compressed bottom particle during impact)
    # would otherwise smear via the 27-node stencil onto equator particles and
    # give them a net inward pull  -  the grid-pollution implosion mechanism.
    # Scale each particle's stress tensor down if its Frobenius norm exceeds
    # MPM_STRESS_CLAMP, leaving all other particles completely unchanged.
    stress_norms = np.linalg.norm(Ps.reshape(len(Ps), -1), axis=1)   # (N,)
    too_large    = stress_norms > MPM_STRESS_CLAMP
    if np.any(too_large):
        scale = np.where(too_large, MPM_STRESS_CLAMP / stress_norms, 1.0)
        Ps    = Ps * scale[:, None, None]

    # --- Step 2a: P2G  -  scatter force contributions to grid nodes ---
    # For each particle n and each of its 27 stencil nodes k:
    #   f_contrib[n, k, :] = -V0 * P_p @ ∇w_ip
    # einsum 'nij,nkj->nki' means: for particle n and stencil node k,
    #   result[k, i] = sum_j  P[i,j] * dw[k,j]   (matrix-vector product P @ ∇w)
    f_contrib = -V0 * np.einsum('nij,nkj->nki', Ps, dw_ip)    # (N, 27, 3)

    # Accumulate force contributions into the flat grid array.
    # np.add.at handles the case where multiple particles write to the same node.
    grid_force = np.zeros((n_grid, 3))
    np.add.at(grid_force, node_idx.ravel(), f_contrib.reshape(-1, 3))

    # --- Step 2b: P2G  -  scatter mass contributions to grid nodes ---
    # Each particle deposits a weighted fraction of its mass onto each nearby node.
    grid_mass = np.zeros(n_grid)
    np.add.at(grid_mass, node_idx.ravel(), (w_ip * PARTICLE_MASS).ravel())

    # --- Step 2c: Divide to get grid accelerations ---
    # a_i = f_i / m_i.  Replace zero-mass nodes with 1.0 to avoid division-by-zero
    # (their force is also zero so the result doesn't matter).
    safe_m     = np.where(grid_mass > 1e-12, grid_mass, 1.0)
    grid_accel = grid_force / safe_m[:, None]                  # (n_grid, 3)

    # --- Step 3: G2P  -  gather accelerations back to particles ---
    # Look up the acceleration at each of the 27 stencil nodes for every particle.
    accel_nodes = grid_accel[node_idx]                          # (N, 27, 3)
    # Weighted average: f_p = m_p * Σ_i  w_ip * a_i
    # einsum 'nk,nki->ni': for particle n, result[i] = sum_k  w[k] * accel[k,i]
    particle_forces = PARTICLE_MASS * np.einsum('nk,nki->ni', w_ip, accel_nodes)
    return particle_forces   # (N, 3)


def compute_apic_step(positions, velocities, Cs, Fs, mat, mu_scale,
                      node_idx, w_ip, dw_ip, dpos_ip, n_grid, dt):
    """
    One APIC (Affine Particle-in-Cell) substep in Total Lagrangian formulation.

    Replaces the separate compute_Fs_batch + compute_mpm_forces calls.
    The grid is fixed in the reference configuration.

    APIC adds an affine velocity matrix C per particle (shape 3×3).
    C approximates the reference-frame velocity gradient ∂v/∂X.

    P2G (Particles → Grid):
      grid_momentum[i] += w_ip * m_p * (v_p + C_p @ dpos_ip)
      grid_mass[i]     += w_ip * m_p
      grid_force[i]    += -V0 * P_p @ ∇w_ip   (stress, same as before)

    Grid update:
      v_grid = (grid_momentum + dt * (grid_force + grid_mass * gravity)) / grid_mass

    G2P (Grid → Particles):
      v_new  = Σ_i  w_ip * v_grid_i
      C_new  = (4/h²) * Σ_i  w_ip * v_grid_i ⊗ dpos_ip_i

    F update (Total Lagrangian  -  additive):
      dF/dt = ∂v/∂X = C_ref  →  F_new = F_old + dt * C_new

    Returns new_vel (N,3), new_C (N,3,3), new_F (N,3,3).
    """
    N = len(positions)

    # --- Compute PK1 stress with magnitude clamp ---
    Ps = compute_pk1_stress(Fs, mat, mu_scale)
    stress_norms = np.linalg.norm(Ps.reshape(N, -1), axis=1)
    too_large = stress_norms > MPM_STRESS_CLAMP
    if np.any(too_large):
        scale = np.where(too_large, MPM_STRESS_CLAMP / stress_norms, 1.0)
        Ps = Ps * scale[:, None, None]

    # --- P2G: scatter APIC momentum, mass, and stress force ---
    grid_mass     = np.zeros(n_grid)
    grid_momentum = np.zeros((n_grid, 3))
    grid_force    = np.zeros((n_grid, 3))

    # APIC momentum: v_p + C_p @ dpos_ip  for each (particle, stencil node)
    # einsum 'nij,nkj->nki': for particle n and node k,  (C @ dpos)[k] = C[i,j]*dpos[k,j]
    Cv          = np.einsum('nij,nkj->nki', Cs, dpos_ip)          # (N, 27, 3)
    mom_contrib = w_ip[:, :, None] * PARTICLE_MASS * (velocities[:, None, :] + Cv)
    mass_contrib = w_ip * PARTICLE_MASS                            # (N, 27)
    np.add.at(grid_momentum, node_idx.ravel(), mom_contrib.reshape(-1, 3))
    np.add.at(grid_mass,     node_idx.ravel(), mass_contrib.ravel())

    # Stress force: same Total Lagrangian formula as before
    f_contrib = -V0 * np.einsum('nij,nkj->nki', Ps, dw_ip)        # (N, 27, 3)
    np.add.at(grid_force, node_idx.ravel(), f_contrib.reshape(-1, 3))

    # --- Grid update: momentum + dt*(force + mass*gravity) then divide by mass ---
    safe_m   = np.where(grid_mass > 1e-12, grid_mass, 1.0)
    grid_vel = (grid_momentum + dt * (grid_force + grid_mass[:, None] * GRAVITY)) / safe_m[:, None]

    # --- G2P: gather new velocity and affine matrix ---
    gv = grid_vel[node_idx]   # (N, 27, 3)  -  grid velocity at each stencil node

    # New particle velocity: weighted sum
    new_vel = np.einsum('nk,nki->ni', w_ip, gv)                    # (N, 3)

    # New affine matrix C_ref = (4/h²) * Σ_k w_k * v_grid_k ⊗ dpos_k
    new_C = (4.0 / MPM_GRID_H**2) * np.einsum('nk,nki,nkj->nij', w_ip, gv, dpos_ip)  # (N,3,3)

    # F update (Total Lagrangian additive): F_new = F_old + dt·C
    # C ≈ ∂v/∂X because our grid is fixed in reference space, so dF/dt = ∂v/∂X → additive.
    # Multiplicative (I + dt·C) @ F is for Updated Lagrangian MPM where C ≈ ∂v/∂x
    # (current-space grid, as Genesis uses)  -  wrong here and causes F to blow up over time.
    new_F = Fs + dt * new_C

    return new_vel, new_C, new_F


# ============================================================
# DIAGNOSTICS  (per-particle scalars for heat-map visualisation)
# ============================================================

def compute_diagnostics(Fs, mat, mu_scale=1.0):
    """
    Compute two per-particle scalars that expose instability before it happens.

    Why these two and not averages?
    --------------------------------
    An implosion is caused by ONE particle whose neighbourhood inverts or whose
    stress spikes.  Averages mask that completely  -  the average stress across
    120 particles barely moves when one particle hits 100x normal stress.
    You need to see WHICH particle is the problem and WHERE it is on the ball.

    Returns
    -------
    stress_norm : (N,)  -  Frobenius norm  ‖P‖  of the PK1 stress tensor.
        This is the total force-generation magnitude at each particle.
        Normal range at rest: ~0.  At bouncing impact: 1–20.
        If you see a bright spot here, that particle is about to cause trouble.

    sigma_min : (N,)  -  smallest singular value of F.
        Singular values of F are the principal stretch ratios.
        sigma = 1.0  → no deformation on that axis (rest state).
        sigma = 0.5  → compressed to half rest length.
        sigma = 0.2  → SVD clamp threshold  -  this particle is on the edge.
        sigma < 0.2  → clamp is activating; these are your problem particles.
        Use the "RdYlGn" colormap: red = danger, green = safe.
    """
    # stress norm: compute P for all particles, then take the matrix Frobenius norm
    # np.linalg.norm on axis=(1,2) computes sqrt(sum of all 9 entries squared) per particle
    Ps          = compute_pk1_stress(Fs, mat, mu_scale)
    stress_norm = np.linalg.norm(Ps.reshape(len(Fs), -1), axis=1)   # (N,)

    # min singular value: SVD returns values in descending order, so index 2 is the smallest
    _, s, _   = np.linalg.svd(Fs)   # s: (N, 3)  each row = [σ_large, σ_mid, σ_small]
    sigma_min = s[:, 2]             # (N,)

    return stress_norm, sigma_min


# ============================================================
# PER-PARTICLE DEFORMATION GRADIENT  (MPM-style, local WLS)
# ============================================================

def build_F_neighbourhood(rest_pos, k=K_PHYS):
    """
    Precompute the data needed to measure how deformed each particle's
    neighbourhood is, using Weighted Least Squares (WLS).

    For each particle i we store:
      nbs[i]      -  indices of the k nearest neighbours
      weights[i]  -  normalised inverse-distance weights (closer = more weight)
      B_mats[i]   -  "rest-state scatter matrix"  B = Σ_k w_k * (X_k-X_i)⊗(X_k-X_i)
      B_invs[i]   -  inverse of B_mats[i]

    B is the weighted sum of outer products of rest-state displacement vectors.
    It encodes the shape of the neighbourhood at rest.
    B_inv is used in compute_Fs_batch to solve for F.
    """
    tree       = cKDTree(rest_pos)
    # query k+1 because the first result is the particle itself (distance 0)
    dists, nbs = tree.query(rest_pos, k=k+1)
    nbs        = nbs[:, 1:]    # drop self
    dists      = dists[:, 1:]

    # inverse-distance weights: closer neighbours contribute more
    w          = 1.0 / (dists + 1e-8)   # 1e-8 prevents division by zero
    w         /= w.sum(axis=1, keepdims=True)  # normalise so weights sum to 1

    N      = len(rest_pos)
    B_mats = np.zeros((N, 3, 3))
    B_invs = np.zeros((N, 3, 3))
    for i in range(N):
        # d: displacement vectors from particle i to each neighbour in rest config
        d         = rest_pos[nbs[i]] - rest_pos[i]         # (k, 3)
        # B = weighted sum of outer products: Σ_k w_k * d_k * d_k^T
        B         = (w[i,:,None] * d).T @ d                # (3, 3)
        B_mats[i] = B
        try:
            # add small regularisation (1e-8 * I) before inverting so nearly-
            # planar neighbourhoods (near the ball surface) don't blow up
            B_invs[i] = np.linalg.inv(B + 1e-8*np.eye(3))
        except:
            B_invs[i] = np.eye(3)   # fallback: treat as undeformed
    return nbs, w, B_mats, B_invs


def compute_Fs_batch(positions, rest_pos, nbs, weights, B_mats, B_invs):
    """
    Compute the deformation gradient F for every particle in one vectorised pass.

    F is a 3×3 matrix that describes how a small piece of material has been
    rotated, stretched, and sheared relative to its rest shape.
      F = I   → no deformation
      det(F) > 1 → volume increased (expansion)
      det(F) < 1 → volume decreased (compression)

    Formula:  F = I + (A - B) @ B⁻¹

    where:
      A[n] = Σ_k  w_k * (x_k - x_n) ⊗ (X_k - X_n)   (current-config scatter)
      B[n] = Σ_k  w_k * (X_k - X_n) ⊗ (X_k - X_n)   (rest-config scatter, precomputed)

    Key insight: using (A - B) instead of just A means F = I exactly when
    the ball moves rigidly (A = B in that case).  This eliminates a constant
    noise term that would otherwise seed exponential growth over many frames.

    This is the FAST vectorised version (used every substep).
    """
    # d_cur[n, k] = current position of neighbour k relative to particle n
    d_cur  = positions[nbs] - positions[:, None, :]   # (N, K, 3)
    # d_rest[n, k] = rest position of neighbour k relative to particle n
    d_rest = rest_pos[nbs]  - rest_pos[:, None, :]    # (N, K, 3)

    # A[n] = Σ_k  w_k * d_cur[k] ⊗ d_rest[k]
    # einsum 'nk,nki,nkj->nij': for particle n,
    #   A[i,j] = sum_k  weight[k] * d_cur[k,i] * d_rest[k,j]
    A      = np.einsum('nk,nki,nkj->nij', weights, d_cur, d_rest)  # (N, 3, 3)

    # deviation from rest: zero for rigid translation, nonzero for deformation
    dA     = A - B_mats                                              # (N, 3, 3)

    eye    = np.eye(3)
    # F = I + dA @ B⁻¹
    # einsum 'nij,njk->nik' is batched matrix multiply: F[n] = dA[n] @ B_inv[n]
    return eye + np.einsum('nij,njk->nik', dA, B_invs)             # (N, 3, 3)


def compute_local_Fs(positions, rest_pos, nbs, weights, B_mats, B_invs):
    """
    Same deformation gradient computation as compute_Fs_batch, but also
    performs a polar decomposition F = R * U on each result.

    R (rotation) : pure rotation matrix  -  which way is the particle oriented?
    U (stretch)  : symmetric positive-definite stretch matrix  -  how squashed/stretched?

    These are used for:
      - Deforming the Gaussian covariances:  C' = F C Fᵀ
      - The shape restoration step (uses R)
      - The HUD display of U_yy (vertical stretch)

    This is the SLOW per-particle version (called once per frame, not per substep).
    """
    N   = len(positions)
    Fs  = np.zeros((N, 3, 3))
    Rs  = np.zeros((N, 3, 3))
    Us  = np.zeros((N, 3, 3))
    eye = np.eye(3)
    for i in range(N):
        d_cur  = positions[nbs[i]] - positions[i]
        d_rest = rest_pos[nbs[i]]  - rest_pos[i]
        A      = (weights[i,:,None] * d_cur).T @ d_rest
        F      = eye + (A - B_mats[i]) @ B_invs[i]
        Fs[i]  = F
        try:
            # polar decomp: scipy returns (R, U) such that F = R @ U
            R, U  = scipy_polar(F)
            Rs[i] = R
            Us[i] = U
        except Exception:
            # if F is degenerate (e.g. particle cluster collapsed), use identity
            Rs[i] = eye
            Us[i] = F
    return Fs, Rs, Us

# ============================================================
# SIMULATION
# ============================================================

# Build the rest positions once at module load time.
# For compare mode we FORCE a rubber-sphere setup for both entities so the
# only difference is the solver  -  that's the entire point of Table 1.
_BUNNY_MESH = None    # set when SHAPE="bunny"; used in run() and shape-swap handler
_CUSTOM_MESH = None   # set when SHAPE="custom"; used in run() and shape-swap handler

if SCENE == "compare":
    SHAPE = "sphere"

# Per-entity stash of PLY-derived rest covariances + render attributes,
# populated by _make_entity_rest when SHAPE == "ply".  _load_material() reads
# rest_covs from here; the renderer (Piece 2) reads opacity + SH from here.
# Per-entity stash of PLY-derived rest covariances + render attributes,
# populated by _make_entity_rest when SHAPE == "ply".  _load_material() reads
# rest_covs from here; the renderer (Piece 2) reads opacity + SH from here.
_PLY_COVS_PER_ENTITY: list = []
_PLY_OPACITY_PER_ENTITY: list = []
_PLY_SHS_PER_ENTITY: list = []


def _make_entity_rest(shape: str, n: int):
    """Sample n particles in entity-local coords (centered at origin).
    Returns (rest_pos, V0, mpm_grid_h, spawn_y_offset, rest_covs).
    `V0` is a scalar OR an (n,) numpy array (when --grid-fill is set).
    `rest_covs` is None (caller uses make_base_covs) OR an (n,3,3) array."""
    global _BUNNY_MESH, _CUSTOM_MESH

    # Helper: run grid-fill for any shape with an interior check + bbox.
    def _gf(interior, bbox_lo, bbox_hi, mpm_h, spawn_y):
        pos, V0a, covs, dx, n_act = fill_shape_grid_based(
            interior, np.asarray(bbox_lo), np.asarray(bbox_hi),
            target_n=n, ppc=PPC,
        )
        pos, V0a, covs = _pad_or_truncate(pos, V0a, covs, n)
        print(f"[grid-fill] shape={shape} dx={dx:.4f} n_actual_before_pad={n_act} "
              f"-> padded/truncated to {n} (ppc={PPC})")
        return pos.astype(np.float32), V0a, mpm_h, spawn_y, covs

    if shape == "ply":
        if not PLY_PATH:
            raise ValueError("SHAPE='ply' requires --ply <path>")
        pts, covs, opac, shs = load_ply_3dgs(PLY_PATH, n)
        _PLY_COVS_PER_ENTITY.append(covs)
        _PLY_OPACITY_PER_ENTITY.append(opac)
        _PLY_SHS_PER_ENTITY.append(shs)
        if PHYSGAUSSIAN_FICUS_MATCH:
            cell = np.floor(pts / 0.04).astype(np.int32)
            _, inverse, counts = np.unique(
                cell, axis=0, return_inverse=True, return_counts=True
            )
            v0 = ((0.04 ** 3) / counts[inverse]).astype(np.float32)
            return pts, v0, 0.04, 0.0, None
        # PLY: covariances are TRAINED  -  we don't override them with grid-fill.
        # V0 still uses the uniform bbox-share formula.
        _ext  = pts.max(axis=0) - pts.min(axis=0)
        _vol  = float(np.prod(_ext))
        return pts, _vol / n, float(max(_ext)) / 12.0, 4.0, None
    if shape == "duck":
        if GRID_FILL_ENABLED:
            return _gf(_duck_interior_check(),
                       [-1.1, -0.75, -1.0], [1.1, 1.55, 1.05], 0.35, 5.5)
        pts = sample_duck_volume(n)
        _v  = 1.0 * 0.65 * 0.85 + 0.45**3 + 0.22*0.12*0.28 + 0.28*0.30*0.22
        _v *= (4.0/3.0) * np.pi
        return pts, _v / n, 0.35, 5.5, None
    if shape == "cuboid":
        if GRID_FILL_ENABLED:
            h = np.asarray(CUBOID_HALF_EXTENTS)
            return _gf(_cuboid_interior_check(CUBOID_HALF_EXTENTS),
                       -h, h, float(min(CUBOID_HALF_EXTENTS) / 4.0), 4.0)
        pts = sample_cuboid_volume(n, CUBOID_HALF_EXTENTS)
        _hx, _hy, _hz = CUBOID_HALF_EXTENTS
        _v = float((2.0 * _hx) * (2.0 * _hy) * (2.0 * _hz)) / n
        return pts, _v, float(min(CUBOID_HALF_EXTENTS) / 4.0), 4.0, None
    if shape == "bunny":
        if not BUNNY_OBJ_PATH.exists():
            raise FileNotFoundError(f"Bunny OBJ not found: {BUNNY_OBJ_PATH}")
        if _BUNNY_MESH is None:
            _BUNNY_MESH = load_centered_mesh(BUNNY_OBJ_PATH, BUNNY_OBJ_SCALE, "bunny")
        if GRID_FILL_ENABLED:
            b   = np.array(_BUNNY_MESH.bounds)
            ext = b[1::2] - b[0::2]
            return _gf(_mesh_interior_check(_BUNNY_MESH),
                       b[0::2], b[1::2],
                       float(max(ext)) / 12.0, float(b[3]) + 4.0)
        p, v, h, sy = _mesh_shape_rest(_BUNNY_MESH, n)
        return p, v, h, sy, None
    if shape == "custom":
        if not CUSTOM_OBJ_PATH:
            raise ValueError("Set CUSTOM_OBJ_PATH when using SHAPE='custom'")
        if _CUSTOM_MESH is None:
            _CUSTOM_MESH = load_centered_mesh(CUSTOM_OBJ_PATH, CUSTOM_OBJ_SCALE, "custom")
        if GRID_FILL_ENABLED:
            b   = np.array(_CUSTOM_MESH.bounds)
            ext = b[1::2] - b[0::2]
            return _gf(_mesh_interior_check(_CUSTOM_MESH),
                       b[0::2], b[1::2],
                       float(max(ext)) / 12.0, float(b[3]) + 4.0)
        p, v, h, sy = _mesh_shape_rest(_CUSTOM_MESH, n)
        return p, v, h, sy, None
    # default: sphere
    if GRID_FILL_ENABLED:
        return _gf(_sphere_interior_check(BALL_RADIUS),
                   [-BALL_RADIUS]*3, [BALL_RADIUS]*3, BALL_RADIUS / 4.0, 4.0)
    pts = sample_sphere_volume(n, BALL_RADIUS)
    return pts, (4.0/3.0 * np.pi * BALL_RADIUS**3) / n, BALL_RADIUS / 4.0, 4.0, None

# Per-entity geometry: each entity independently sampled in its own local frame
_PER_ENTITY_REST       = []
_PER_ENTITY_V0         = []  # scalar OR (N,) numpy array (when --grid-fill)
_PER_ENTITY_GRIDH      = []
_PER_ENTITY_SPAWN_Y    = []
_PER_ENTITY_REST_COVS  = []  # None (use make_base_covs) OR (N,3,3) array
for _ent_idx in range(N_ENTITIES):
    _shape_for_entity = ENTITY_SHAPES[_ent_idx] if ENTITY_SHAPES is not None else SHAPE
    _r, _v0, _h, _sy, _covs = _make_entity_rest(_shape_for_entity, N_PER_ENTITY)
    if FICUS_DROP_COMPARE and _ent_idx == 1:
        # Invert only the second copy: pot/base moves to high-y and the jelly
        # canopy becomes the lower contact side.  Rotate covariance accordingly
        # so trained anisotropic splats follow the flipped geometry.
        _r = _r.copy()
        _r[:, 1] *= -1.0
        if SHAPE == "ply" and _ent_idx < len(_PLY_COVS_PER_ENTITY):
            _flip_y = np.diag([1.0, -1.0, 1.0]).astype(np.float32)
            _PLY_COVS_PER_ENTITY[_ent_idx] = np.einsum(
                "ij,njk,lk->nil", _flip_y, _PLY_COVS_PER_ENTITY[_ent_idx], _flip_y
            ).astype(np.float32)
        if _covs is not None:
            _flip_y = np.diag([1.0, -1.0, 1.0]).astype(np.float32)
            _covs = np.einsum("ij,njk,lk->nil", _flip_y, _covs, _flip_y).astype(np.float32)
    _PER_ENTITY_REST.append(_r.astype(np.float32))
    if isinstance(_v0, np.ndarray):
        _PER_ENTITY_V0.append(_v0.astype(np.float32))
    else:
        _PER_ENTITY_V0.append(float(_v0))
    _PER_ENTITY_GRIDH.append(float(_h))
    _PER_ENTITY_SPAWN_Y.append(float(_sy))
    _PER_ENTITY_REST_COVS.append(_covs)

REST_POS    = np.concatenate(_PER_ENTITY_REST, axis=0)   # (N_PARTICLES, 3) in entity-local coords
V0          = _PER_ENTITY_V0[0]                           # same per entity in compare mode
# Use the FINEST grid_h across entities for the world grid spacing.  Otherwise
# a coarse-grained entity (e.g. cuboid with dx=0.5) makes the floor BC band
# stretch into other bodies, zeroing their downward velocity prematurely.
MPM_GRID_H  = (0.04 if PHYSGAUSSIAN_FICUS_MATCH else
               (float(_CLI.grid_dx) if _CLI.grid_dx is not None
               else (min(_PER_ENTITY_GRIDH) if SCENE == "collide" else _PER_ENTITY_GRIDH[0]))
              )
MAX_SPEED   = 0.5 * MPM_GRID_H / (DT / SUBSTEPS)   # recompute CFL with actual grid spacing
# Per-entity spawn world positions.  In compare mode put the entities side by
# side along x so their grids don't touch.  In single mode use the original
# centered spawn so existing scenes are unchanged.
if FICUS_DROP_COMPARE or FICUS_METAL_COMPARE:
    SPAWN_POSITIONS = [
        np.array([-1.65, 4.0, 0.0]),
        np.array([+1.65, 4.0, 0.0]),
    ]
elif SCENE == "compare":
    SPAWN_POSITIONS = [
        np.array([-2.0, _PER_ENTITY_SPAWN_Y[0], 0.0]),
        np.array([+2.0, _PER_ENTITY_SPAWN_Y[1], 0.0]),
    ]
elif SCENE == "collide":
    # Multi-material collide: spawn positions come from _COLLIDE_ENTITIES spec.
    SPAWN_POSITIONS = [np.array(e[2], dtype=np.float64) for e in _COLLIDE_ENTITIES]
else:
    SPAWN_POSITIONS = [np.array([0.0, _PER_ENTITY_SPAWN_Y[0], 0.0])]
# Backward-compat alias  -  old code paths still read BALL_START
BALL_START  = SPAWN_POSITIONS[0]


# ============================================================
# TAICHI GPU BACKEND
# Fields and kernels that replace the NumPy hot path.
# All physics runs on the RTX 4070; only per-frame rendering
# data (positions, Us, covs) is copied back to CPU.
# ============================================================

_MAX_GRID = 32768   # generous upper bound for MPM grid nodes

# PBF constants  -  fluid only (Position-Based Fluids, Macklin & Müller 2013).
# Uses the paper's cubic SPH kernel.  Brute-force O(N²) at N=120  -  trivial on GPU.
_PBF_H    = 0.6          # smoothing radius (m)
_PBF_H2   = _PBF_H ** 2  # h² cached
_PBF_ITER = 10            # constraint projection iterations per substep
_PBF_EPS  = 600.0         # relaxation term in lambda denominator (prevents div/0, adds softness)
_PBF_XSPH = 0.05          # XSPH viscosity coefficient (smooths velocity field)
_PBF_D0   = _PBF_H * 0.4  # distance constraint threshold (minimum particle separation)

# Spatial hash grid for PBF neighbour queries.
# Cell side = _PBF_H so each cell maps to exactly one smoothing sphere.
# 27-cell stencil then covers all possible neighbours.
_PBF_GRID_ORIGIN  = -6.0          # world-space origin (all three axes)
_PBF_GRID_N       = 24            # cells per axis: 24x0.6 = 14.4 m span (-6..8.4)
_PBF_MAX_PER_CELL = 24            # hard cap; ~6 expected for N=120 in a 1-m ball
_PBF_TOTAL_CELLS  = _PBF_GRID_N ** 3

# --- Particle state ---
_ti_pos  = ti.Vector.field(3, dtype=ti.f32, shape=N_PARTICLES)
_ti_vel  = ti.Vector.field(3, dtype=ti.f32, shape=N_PARTICLES)
_ti_C    = ti.Matrix.field(3, 3, dtype=ti.f32, shape=N_PARTICLES)
_ti_F    = ti.Matrix.field(3, 3, dtype=ti.f32, shape=N_PARTICLES)
_ti_R    = ti.Matrix.field(3, 3, dtype=ti.f32, shape=N_PARTICLES)
_ti_U    = ti.Matrix.field(3, 3, dtype=ti.f32, shape=N_PARTICLES)

# Textbook continuum-mechanics state (derived diagnostics, updated each frame).
# J      -  Jacobian, J=det(F); volume ratio.  J=1 rest, J=0.5 half-volume, J=2 doubled.
# rho_t  -  current mass density = m / (J * V0).  Tracks conservation of mass.
# C_cg   -  Right Cauchy-Green tensor = F^T·F.  The rotation-invariant strain object
#         that any frame-indifferent constitutive law should depend on.
# T      -  Cauchy stress = (1/J) · P · F^T (with P from the active constitutive law).
#         The "true" stress in current configuration  -  force per unit current area.
_ti_J     = ti.field(ti.f32, shape=N_PARTICLES)
_ti_rho_t = ti.field(ti.f32, shape=N_PARTICLES)
_ti_C_cg  = ti.Matrix.field(3, 3, dtype=ti.f32, shape=N_PARTICLES)
_ti_T     = ti.Matrix.field(3, 3, dtype=ti.f32, shape=N_PARTICLES)

# --- Rest-pose constants (re-uploaded on shape swap) ---
_ti_rest = ti.Vector.field(3, dtype=ti.f32, shape=N_PARTICLES)  # rest position x₀
_ti_rcov = ti.Matrix.field(3, 3, dtype=ti.f32, shape=N_PARTICLES)  # rest covariance Σ₀
_ti_v0   = ti.field(ti.f32, shape=N_PARTICLES)                   # rest volume V₀ per particle
_ti_m    = ti.field(ti.f32, shape=N_PARTICLES)                   # mass = density * V₀ per particle

# --- Render outputs (written by GPU, read by CPU each frame) ---
_ti_covs = ti.Matrix.field(3, 3, dtype=ti.f32, shape=N_PARTICLES)
_ti_sn   = ti.field(ti.f32, shape=N_PARTICLES)   # stress-norm diagnostics
_ti_sm   = ti.field(ti.f32, shape=N_PARTICLES)   # sigma_min diagnostics
_ti_Fp   = ti.Matrix.field(3, 3, dtype=ti.f32, shape=N_PARTICLES)  # plastic F  -  permanent deformation
# Fracture state  -  both permanent (ratchet only).
_ti_damage      = ti.field(ti.f32, shape=N_PARTICLES)                # damage scalar [0,1], permanent
# Per-bond intact flag (1 = active, 0 = broken).  Indexed (particle, k_neighbour);
# uses the same K_PHYS layout as _ti_wnb_tl / _ti_ww_tl in solver_tl.
_ti_bond_intact = ti.field(ti.i32, shape=(N_PARTICLES, K_PHYS))

# --- Shape restore (solid only)  -  COM + H accumulators written by GPU; R written by CPU SVD ---
_ti_shape_buf = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())   # H = Σ rest_p ⊗ (pos_p − com)
_ti_shape_com = ti.Vector.field(3, dtype=ti.f32, shape=())       # center of mass
_ti_shape_R   = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())    # best-fit rotation (filled by CPU SVD)

# --- PBF scratch (fluid only) ---
# Gripper plate  -  y position of the bottom face, updated from Python each frame
_ti_gripper_y = ti.field(ti.f32, shape=())   # scalar field, read in _ti_g2p
_ti_bottom_gripper_y = ti.field(ti.f32, shape=())
_ti_side_plate_lx = ti.field(ti.f32, shape=())
_ti_side_plate_rx = ti.field(ti.f32, shape=())
_ti_gripper_y[None] = 9999.0                 # far above scene  -  inactive by default

# Hammer  -  sphere collider driven by Xbox controller
HAMMER_RADIUS = 0.22   # metres  -  compact striking face
HAMMER_K      = 2500.0 # contact stiffness
_ti_hammer_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
_ti_hammer_vel = ti.Vector.field(3, dtype=ti.f32, shape=())
_ti_bottom_gripper_y[None] = -9999.0
_ti_side_plate_lx[None] = -9999.0
_ti_side_plate_rx[None] = 9999.0
_ti_hammer_pos[None] = ti.Vector([0.0, 9999.0, 0.0])  # inactive above scene
_ti_hammer_vel[None] = ti.Vector([0.0, 0.0, 0.0])

# ============================================================
# Hand-tracker collider state (Tier 3: capsule-skeleton hand)
# ============================================================
# Up to 2 hands x 21 landmarks each = 42 joint positions.  Indexed as
#   hand 0 -> [0, 21), hand 1 -> [21, 42).
# A hand is "active" only when MediaPipe has a detection this frame; idle
# hands park their landmarks far above the scene so the contact test is a
# cheap no-op (matches the hammer's parked-at-y=9999 idiom).
N_HAND_PTS_PER  = 21
N_HANDS_MAX     = 2
N_HAND_PTS      = N_HAND_PTS_PER * N_HANDS_MAX

_ti_hand_pts    = ti.Vector.field(3, dtype=ti.f32, shape=N_HAND_PTS)
_ti_hand_pts_v  = ti.Vector.field(3, dtype=ti.f32, shape=N_HAND_PTS)   # finite-diff velocity per joint
_ti_hand_active = ti.field(ti.i32, shape=N_HANDS_MAX)                  # 0 / 1

# Park all joints high above the scene initially
for _i in range(N_HAND_PTS):
    _ti_hand_pts[_i]   = ti.Vector([0.0, 9999.0, 0.0])
    _ti_hand_pts_v[_i] = ti.Vector([0.0, 0.0, 0.0])
for _i in range(N_HANDS_MAX):
    _ti_hand_active[_i] = 0

# MediaPipe hand connections (23 bones per hand).  Each pair is (joint_a, joint_b)
# within the 21-landmark layout: 0=wrist, thumb 1-4, index 5-8, middle 9-12,
# ring 13-16, pinky 17-20.  Last 3 are palm cross-knuckle links.
HAND_CONNECTIONS = (
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
)
N_HAND_BONES_PER  = len(HAND_CONNECTIONS)         # 23 bones per hand
N_HAND_BONES      = N_HAND_BONES_PER * N_HANDS_MAX

# Flat (bone_idx -> (joint_a, joint_b) in global 42-joint index space) table for kernels.
# Read in @ti.func _hand_contact() during G2P.
_HAND_BONE_AB_NP = np.zeros((N_HAND_BONES, 2), dtype=np.int32)
for _h in range(N_HANDS_MAX):
    for _b, (_a, _bb) in enumerate(HAND_CONNECTIONS):
        _HAND_BONE_AB_NP[_h * N_HAND_BONES_PER + _b, 0] = _h * N_HAND_PTS_PER + _a
        _HAND_BONE_AB_NP[_h * N_HAND_BONES_PER + _b, 1] = _h * N_HAND_PTS_PER + _bb
_ti_hand_bone_ab = ti.field(ti.i32, shape=(N_HAND_BONES, 2))
_ti_hand_bone_ab.from_numpy(_HAND_BONE_AB_NP)


def image_to_world(landmarks_normalized: np.ndarray) -> np.ndarray:
    """Map MediaPipe normalised image coords (x,y in [0,1], z relative) to
    world coords.

    - x: un-mirrored so user's right hand appears on world-right
    - y: image-top -> world-top
    - z: MediaPipe's relative-depth z (origin at wrist, more negative =
      closer to camera) scaled by HAND_Z_SCALE.  Per-landmark depth is
      preserved so fingertips can reach in/out of the scene independently.

    landmarks_normalized: (21, 3) array
    returns: (21, 3) world-space coords
    """
    w = np.empty_like(landmarks_normalized)
    # Image x in [0,1]: 0 = image-left = USER'S RIGHT side (camera mirrors).
    # Un-mirror so reaching right maps to +world_x.
    w[:, 0] = (0.5 - landmarks_normalized[:, 0]) * 2.0 * HAND_X_HALF
    # Image y in [0,1]: 0 = top (world high), 1 = bottom (world low).
    w[:, 1] = HAND_Y_MAX - landmarks_normalized[:, 1] * (HAND_Y_MAX - HAND_Y_MIN)
    # MediaPipe z: 0 at wrist, negative toward camera.  Scale + sign-flip so
    # reaching toward the user moves the hand to +world_z.
    w[:, 2] = HAND_Z_CENTER + landmarks_normalized[:, 2] * HAND_Z_SCALE
    return w

_ti_mu      = ti.field(ti.f32, shape=N_PARTICLES)              # shear modulus μ (thermally softened per frame)
_ti_mu0     = ti.field(ti.f32, shape=N_PARTICLES)              # base shear modulus μ₀ at reference temperature
_ti_lam     = ti.field(ti.f32, shape=N_PARTICLES)              # Lamé λ (bulk)
_ti_damping = ti.field(ti.f32, shape=N_PARTICLES)              # velocity decay rate α (1/s)
_ti_yield   = ti.field(ti.f32, shape=N_PARTICLES)              # yield stress threshold
_ti_relax   = ti.field(ti.f32, shape=N_PARTICLES)              # Fp relaxation time constant (s)
_ti_plastic_j2 = ti.field(ti.i32, shape=N_PARTICLES)            # 1 = J2 (von Mises) yield on deviatoric PK1
_ti_tension_cap = ti.field(ti.f32, shape=N_PARTICLES)          # max principal stretch σ_i ≤ 1+cap (0 = off)
_ti_sr_gamma = ti.field(ti.f32, shape=N_PARTICLES)             # μ_eff = μ(1+γ‖C‖) for shear stiffening
_ti_dp_alpha = ti.field(ti.f32, shape=N_PARTICLES)             # Drucker-Prager friction cone α = sin(φ)/(√3·cos(φ))
_ti_model    = ti.field(ti.i32, shape=N_PARTICLES)             # per-particle constitutive law: 0=corotated, 1=fluid, 2=sand, 3=stvk (metal)
_ti_hardening_xi = ti.field(ti.f32, shape=N_PARTICLES)         # per-particle hardening rate: yield += 2μ·xi·Δγ on plastic flow
# ---- Thermodynamics (Stomakhin 2014)  -  per-particle heat + phase state ----
_ti_temp   = ti.field(ti.f32, shape=N_PARTICLES)               # temperature T (°C-ish, arbitrary units)
_ti_heat_U = ti.field(ti.f32, shape=N_PARTICLES)               # latent-heat buffer U ∈ [0, L]; fills to melt
_ti_phase  = ti.field(ti.i32, shape=N_PARTICLES)               # 0 = solid, 1 = fluid (hysteresis state)
_ti_kappa_p = ti.field(ti.f32, shape=N_PARTICLES)              # per-particle bulk modulus for fluid pressure (set on melt)
_ti_rho    = ti.field(ti.f32, shape=N_PARTICLES)               # per-particle density
_ti_lambda = ti.field(ti.f32, shape=N_PARTICLES)               # Lagrange multipliers
_ti_dx     = ti.Vector.field(3, dtype=ti.f32, shape=N_PARTICLES)  # position corrections
_ti_pred   = ti.Vector.field(3, dtype=ti.f32, shape=N_PARTICLES)  # predicted positions
_ti_benchmark_P = ti.Matrix.field(3, 3, dtype=ti.f32, shape=N_PARTICLES)
_ti_benchmark_clamp_mask = ti.field(ti.i32, shape=N_PARTICLES)
_ti_benchmark_clamp_pos = ti.Vector.field(3, dtype=ti.f32, shape=N_PARTICLES)
_ti_benchmark_force_mask = ti.field(ti.i32, shape=N_PARTICLES)
_pbf_cell_count = ti.field(ti.i32, shape=_PBF_TOTAL_CELLS)
_pbf_cell_plist = ti.field(ti.i32, shape=(_PBF_TOTAL_CELLS, _PBF_MAX_PER_CELL))

# --- MPM grid + stencil + WLS state moved to solver_tl.py / solver_ul.py ---
# Each solver module owns the fields it scribbles on each substep, so the
# TL (rest-frame) and UL (world-frame) grids coexist without aliasing.


def _ti_upload_stencil(node_idx, w_ip, dw_ip, dpos_ip,
                       wls_nbs, wls_weights, wls_B_mats, wls_B_invs,
                       rest_pos, slice_start=0):
    """Copy precomputed NumPy neighbourhood arrays to the TL solver's GPU fields.
    `slice_start` is the particle offset for the entity being initialised
    (entity 0 starts at 0; entity 1  -  if it ever uses TL  -  at N_PER_ENTITY).
    Also writes rest position and per-particle V0/mass into the shared
    particle struct (these are NOT solver-specific)."""
    import solver_tl as _tl                                  # lazy: avoid circular import
    _tl.upload_stencil(node_idx, w_ip, dw_ip, dpos_ip, start=slice_start)
    _tl.upload_wls(wls_nbs, wls_weights, wls_B_mats, wls_B_invs, start=slice_start)

    # Particle-level uploads  -  these live in gaussian_mesh_poc because they're
    # shared across solvers (UL also reads _ti_rest, _ti_v0, _ti_m).
    N = node_idx.shape[0]
    sl = slice(slice_start, slice_start + N)
    # Read-modify-write the sliced region of the particle fields.
    cur_rest = _ti_rest.to_numpy()
    cur_v0   = _ti_v0.to_numpy()
    cur_m    = _ti_m.to_numpy()
    cur_rest[sl] = rest_pos.astype(np.float32)
    # V0 may be a scalar (uniform per-entity, default) or a per-particle array
    # of length N (when --grid-fill is set).  Both paths broadcast into the slice.
    if isinstance(V0, np.ndarray):
        cur_v0[sl] = V0[:N].astype(np.float32)
    else:
        cur_v0[sl] = np.full(N, V0, dtype=np.float32)
    cur_m[sl]    = np.full(N, PARTICLE_MASS, dtype=np.float32)
    _ti_rest.from_numpy(cur_rest)
    _ti_v0.from_numpy(cur_v0)
    _ti_m.from_numpy(cur_m)


# ---- Constitutive law functions (called from kernels) ----

@ti.func
def _pk1_corotated(F: ti.template(), mu_eff: ti.f32, lam: ti.f32):
    """First Piola-Kirchhoff stress for fixed-corotated elasticity.
        P = 2μ·(F − R) + λ·(J−1)·J·F^{−T}
    R from polar decomp F = R·S.  Stomakhin 2013 sign-flip applied when
    det(F) < 0 so R is a proper rotation under inversion.

    NO magnitude clamp on singular values  -  the earlier [0.2, 5.0] clamp was
    a workaround for soft materials where extreme deformation produced
    unphysical stress; with PhysGaussian-scale stiffness the natural elastic
    response keeps F bounded, and clamping just underestimates restoring
    force on impact (less rebound).  Tiny epsilon on 1/σ guards against
    division by zero in the inverse term."""
    U, sigma, V = ti.svd(F)
    if (U @ V.transpose()).determinant() < 0:
        V[0, 2] = -V[0, 2]
        V[1, 2] = -V[1, 2]
        V[2, 2] = -V[2, 2]
        sigma[2, 2] = -sigma[2, 2]
    R  = U @ V.transpose()
    s0 = sigma[0, 0]
    s1 = sigma[1, 1]
    s2 = sigma[2, 2]
    J  = s0 * s1 * s2
    # Safe inverse  -  only the epsilon is to prevent 1/0 if a singular value
    # collapses exactly to zero.  Real-valued σ are positive after sign-flip.
    inv0 = 1.0 / ti.max(ti.abs(s0), 1e-8)
    inv1 = 1.0 / ti.max(ti.abs(s1), 1e-8)
    inv2 = 1.0 / ti.max(ti.abs(s2), 1e-8)
    Fi = V @ ti.Matrix([[inv0, 0.0,  0.0 ],
                        [0.0,  inv1, 0.0 ],
                        [0.0,  0.0,  inv2]]) @ U.transpose()
    return 2.0 * mu_eff * (F - R) + lam * (J - 1.0) * J * Fi.transpose()


@ti.func
def _pk1_fluid(F: ti.template(), kappa: ti.f32, mu_reg: ti.f32):
    U, sigma, V = ti.svd(F)
    s0 = ti.min(ti.max(sigma[0, 0], 0.2), 5.0)
    s1 = ti.min(ti.max(sigma[1, 1], 0.2), 5.0)
    s2 = ti.min(ti.max(sigma[2, 2], 0.2), 5.0)
    Sc = ti.Matrix([[s0,  0.0, 0.0],
                    [0.0, s1,  0.0],
                    [0.0, 0.0, s2 ]])
    Fc = U @ Sc @ V.transpose()
    J  = s0 * s1 * s2
    Fi = V @ ti.Matrix([[1.0/s0, 0.0,    0.0   ],
                        [0.0,    1.0/s1, 0.0   ],
                        [0.0,    0.0,    1.0/s2]]) @ U.transpose()
    return kappa * (J - 1.0) * J * Fi.transpose() + mu_reg * Fc


@ti.func
def _mu_eff_sr(p: ti.i32, mu_scale: ti.f32) -> ti.f32:
    """Shear modulus with optional strain-rate stiffening: μ_eff = μ (1 + γ ‖C‖)."""
    m = _ti_mu[p] * mu_scale
    g = _ti_sr_gamma[p]
    if g > 1e-12:
        m *= (1.0 + g * _ti_C[p].norm())
    return m


@ti.func
def _pk1_sand(F: ti.template(), mu_eff: ti.f32, lam: ti.f32):
    """Neo-Hookean log-strain PK1 stress for Drucker-Prager materials (sand/clay).
    Operates in Hencky strain space ε = log(σ) where σ are singular values of F.
    Matches PhysGaussian's kirchoff_stress_drucker_prager (mpm_utils.py:60)
    but returns PK1 = τ·F^{-T} rather than Kirchhoff τ."""
    U, sigma, V = ti.svd(F)
    if (U @ V.transpose()).determinant() < 0:
        V[0, 2] = -V[0, 2]
        V[1, 2] = -V[1, 2]
        V[2, 2] = -V[2, 2]
        sigma[2, 2] = -sigma[2, 2]
    s0 = ti.max(ti.abs(sigma[0, 0]), 1e-8)
    s1 = ti.max(ti.abs(sigma[1, 1]), 1e-8)
    s2 = ti.max(ti.abs(sigma[2, 2]), 1e-8)
    eps0 = ti.log(s0)
    eps1 = ti.log(s1)
    eps2 = ti.log(s2)
    tr_eps = eps0 + eps1 + eps2
    p0 = (2.0 * mu_eff * eps0 + lam * tr_eps) / s0
    p1 = (2.0 * mu_eff * eps1 + lam * tr_eps) / s1
    p2 = (2.0 * mu_eff * eps2 + lam * tr_eps) / s2
    P_diag = ti.Matrix([[p0, 0.0, 0.0],
                        [0.0, p1, 0.0],
                        [0.0, 0.0, p2]])
    return U @ P_diag @ V.transpose()


@ti.func
def _pk1_stvk(F: ti.template(), mu_eff: ti.f32, lam: ti.f32):
    """St Venant-Kirchhoff log-strain stress (Hencky)  -  for metals using
    log-strain J2 plasticity.  Mathematically identical to _pk1_sand but
    kept as a separate name so the metal dispatch is semantically clear."""
    return _pk1_sand(F, mu_eff, lam)


@ti.kernel
def _ti_benchmark_eval_pk1(
    count: ti.i32,
    model: ti.i32,
    mu: ti.f32,
    lam: ti.f32,
    kappa: ti.f32,
):
    """Thin validation hook that executes the production constitutive functions."""
    for p in range(count):
        F = _ti_F[p]
        if model == 0:
            _ti_benchmark_P[p] = _pk1_corotated(F, mu, lam)
        elif model == 1:
            _ti_benchmark_P[p] = _pk1_fluid(F, kappa, mu)
        elif model == 2:
            _ti_benchmark_P[p] = _pk1_sand(F, mu, lam)
        else:
            _ti_benchmark_P[p] = _pk1_stvk(F, mu, lam)


@ti.kernel
def _ti_benchmark_enforce_clamp(count: ti.i32):
    """Restore benchmark-clamped particles after a production solver substep."""
    for p in range(count):
        if _ti_benchmark_clamp_mask[p] != 0:
            _ti_pos[p] = _ti_benchmark_clamp_pos[p]
            _ti_vel[p] = ti.Vector([0.0, 0.0, 0.0])
            _ti_C[p] = ti.Matrix.zero(ti.f32, 3, 3)
            _ti_F[p] = ti.Matrix.identity(ti.f32, 3)
            _ti_Fp[p] = ti.Matrix.identity(ti.f32, 3)


@ti.kernel
def _ti_benchmark_apply_force(
    count: ti.i32,
    force_x: ti.f32,
    force_y: ti.f32,
    force_z: ti.f32,
    dt: ti.f32,
):
    for p in range(count):
        if _ti_benchmark_force_mask[p] != 0:
            force = ti.Vector([force_x, force_y, force_z])
            _ti_vel[p] += dt * force / ti.max(_ti_m[p], 1e-12)


def benchmark_evaluate_pk1(Fs, *, model, mu, lam=0.0, kappa=0.0):
    """Evaluate PK1 using the same Taichi functions called by UL P2G."""
    Fs = np.asarray(Fs, dtype=np.float32)
    if Fs.ndim != 3 or Fs.shape[1:] != (3, 3):
        raise ValueError("Fs must have shape (N, 3, 3)")
    if len(Fs) > N_PARTICLES:
        raise ValueError(f"Need {len(Fs)} fields, simulator allocated {N_PARTICLES}")
    current = _ti_F.to_numpy()
    current[:len(Fs)] = Fs
    _ti_F.from_numpy(current)
    _ti_benchmark_eval_pk1(
        len(Fs), int(model), float(mu), float(lam), float(kappa)
    )
    return _ti_benchmark_P.to_numpy()[:len(Fs)].copy()


def benchmark_upload_particles(positions, volumes, material):
    """Upload a deterministic particle block while retaining production fields."""
    positions = np.asarray(positions, dtype=np.float32)
    volumes = np.asarray(volumes, dtype=np.float32)
    count = len(positions)
    if positions.shape != (count, 3) or volumes.shape != (count,):
        raise ValueError("positions must be (N,3) and volumes must be (N,)")
    if count != N_PARTICLES:
        raise ValueError(
            f"Benchmark block has {count} particles; import allocated {N_PARTICLES}"
        )

    model_name = material["model"]
    model = {"corotated": 0, "fluid": 1, "sand": 2, "stvk": 3}[model_name]
    density = float(material["density"])
    eye = np.tile(np.eye(3, dtype=np.float32), (count, 1, 1))
    zero_vec = np.zeros((count, 3), dtype=np.float32)
    zero_mat = np.zeros((count, 3, 3), dtype=np.float32)

    _ti_rest.from_numpy(positions)
    _ti_pos.from_numpy(positions)
    _ti_vel.from_numpy(zero_vec)
    _ti_C.from_numpy(zero_mat)
    _ti_F.from_numpy(eye)
    _ti_Fp.from_numpy(eye)
    _ti_v0.from_numpy(volumes)
    _ti_m.from_numpy(density * volumes)
    _ti_mu.from_numpy(np.full(count, float(material.get("mu", 0.0)), np.float32))
    _ti_mu0.from_numpy(np.full(count, float(material.get("mu", 0.0)), np.float32))
    _ti_lam.from_numpy(np.full(count, float(material.get("lam", 0.0)), np.float32))
    _ti_damping.from_numpy(
        np.full(count, float(material.get("damping_rate", 0.0)), np.float32)
    )
    _ti_yield.from_numpy(
        np.full(count, float(material.get("yield_stress", 1e9)), np.float32)
    )
    _ti_relax.from_numpy(
        np.full(count, float(material.get("relax_time", 0.0)), np.float32)
    )
    _ti_plastic_j2.from_numpy(
        np.full(count, int(material.get("plastic_j2", 0)), np.int32)
    )
    _ti_tension_cap.from_numpy(
        np.full(count, float(material.get("tension_stretch_cap", 0.0)), np.float32)
    )
    _ti_sr_gamma.from_numpy(
        np.full(count, float(material.get("strain_rate_gamma", 0.0)), np.float32)
    )
    phi = np.radians(float(material.get("friction_angle", 0.0)))
    alpha = np.sin(phi) / (np.sqrt(3.0) * np.cos(phi)) if phi > 0.0 else 0.0
    _ti_dp_alpha.from_numpy(np.full(count, alpha, np.float32))
    _ti_model.from_numpy(np.full(count, model, np.int32))
    _ti_hardening_xi.from_numpy(
        np.full(count, float(material.get("hardening_xi", 0.0)), np.float32)
    )
    _ti_kappa_p.from_numpy(
        np.full(
            count,
            float(material.get("kappa", 0.0)) if model == 1 else 0.0,
            np.float32,
        )
    )
    _ti_damage.from_numpy(np.zeros(count, np.float32))
    _ti_benchmark_clamp_mask.from_numpy(np.zeros(count, np.int32))
    _ti_benchmark_clamp_pos.from_numpy(positions)
    _ti_benchmark_force_mask.from_numpy(np.zeros(count, np.int32))
    return model


def benchmark_set_clamp(mask, positions=None):
    """Configure the benchmark clamp mask and target positions."""
    mask = np.asarray(mask, dtype=np.int32)
    if mask.shape != (N_PARTICLES,):
        raise ValueError(f"clamp mask must have shape ({N_PARTICLES},)")
    _ti_benchmark_clamp_mask.from_numpy(mask)
    targets = _ti_pos.to_numpy() if positions is None else np.asarray(
        positions, dtype=np.float32
    )
    _ti_benchmark_clamp_pos.from_numpy(targets)


def benchmark_enforce_clamp():
    _ti_benchmark_enforce_clamp(N_PARTICLES)


def benchmark_set_force_mask(mask):
    mask = np.asarray(mask, dtype=np.int32)
    if mask.shape != (N_PARTICLES,):
        raise ValueError(f"force mask must have shape ({N_PARTICLES},)")
    _ti_benchmark_force_mask.from_numpy(mask)


def benchmark_apply_force(force, dt):
    force = np.asarray(force, dtype=np.float32)
    _ti_benchmark_apply_force(
        N_PARTICLES, float(force[0]), float(force[1]), float(force[2]), float(dt)
    )


def benchmark_collect_state(*, kappa=0.0, mu_scale=1.0):
    """Collect production particle state and refresh continuum diagnostics."""
    _ti_continuum_state_update(
        0, N_PARTICLES, float(kappa), float(mu_scale), 0
    )
    return {
        "x": _ti_pos.to_numpy().copy(),
        "v": _ti_vel.to_numpy().copy(),
        "F": _ti_F.to_numpy().copy(),
        "Fp": _ti_Fp.to_numpy().copy(),
        "C": _ti_C.to_numpy().copy(),
        "J": _ti_J.to_numpy().copy(),
        "rho": _ti_rho_t.to_numpy().copy(),
        "stress": _ti_T.to_numpy().copy(),
        "yield_stress": _ti_yield.to_numpy().copy(),
    }


@ti.func
def _pbf_W(r: ti.f32) -> ti.f32:
    """Cubic SPH kernel (Gaussian Splashing / Müller cubic).
    W(r,h) = (8/πh³) · [6q²(q−1)+1]   0 ≤ q ≤ 0.5
           = (8/πh³) · 2(1−q)³         0.5 < q ≤ 1
           = 0                          q > 1
    where q = r/h.  C¹ continuous, compact support radius h."""
    q      = r / _PBF_H
    coeff  = 8.0 / (ti.math.pi * _PBF_H * _PBF_H * _PBF_H)
    result = 0.0
    if q <= 0.5:
        result = coeff * (6.0 * q * q * (q - 1.0) + 1.0)
    elif q <= 1.0:
        t = 1.0 - q
        result = coeff * 2.0 * t * t * t
    return result


@ti.func
def _pbf_gradW(xij: ti.math.vec3, r: ti.f32) -> ti.math.vec3:
    """Gradient of cubic SPH kernel w.r.t. xᵢ (points from j→i).
    ∇W = (48/πh⁵)·(3q−2)·xij            0 ≤ q ≤ 0.5
       = −(48/πh⁵)·((1−q)²/q)·xij       0.5 < q ≤ 1"""
    q      = r / _PBF_H
    coeff  = 48.0 / (ti.math.pi * _PBF_H * _PBF_H * _PBF_H * _PBF_H * _PBF_H)
    result = ti.Vector([0.0, 0.0, 0.0])
    if r > 1e-6:
        if q <= 0.5:
            result = coeff * (3.0 * q - 2.0) * xij
        elif q <= 1.0:
            t = (1.0 - q)
            result = -coeff * (t * t / q) * xij
    return result


# ---- GPU kernels ----
# Solver-specific kernels (TL P2G/G2P/grid_update/WLS, UL P2G/G2P/grid_update,
# UL world-grid rebuild) live in solver_tl.py and solver_ul.py.  This module
# keeps only the kernels that are solver-agnostic: polar decomposition,
# plasticity update, covariance update, diagnostics, shape restore, skinning.


@ti.kernel
def _ti_fluid_reset_F():
    eye = ti.Matrix.identity(ti.f32, 3)
    for p in range(N_PARTICLES):
        _ti_F[p] = eye


@ti.kernel
def _ti_polar_decomp():
    for p in range(N_PARTICLES):
        U, sigma, V = ti.svd(_ti_F[p])
        _ti_R[p] = U @ V.transpose()
        _ti_U[p] = V @ sigma @ V.transpose()   # symmetric stretch


@ti.kernel
def _ti_continuum_state_update(start: ti.i32, end: ti.i32,
                               kappa: ti.f32, mu_scale: ti.f32, model: ti.i32):
    """Refresh the four textbook-continuum-mechanics diagnostic fields:
    J (Jacobian), rho_t (current density), C_cg (Right Cauchy-Green), and
    T (Cauchy stress).  Uses the active constitutive law to recompute PK1
    and then converts to Cauchy via T = (1/J) P F^T.  Diagnostic only  - 
    not consumed by the physics loop, which still uses PK1/Kirchhoff in P2G."""
    for p in range(start, end):
        F = _ti_F[p]
        J = F.determinant()
        # Volume ratio + current density (m / current_volume)
        _ti_J[p]     = J
        _ti_rho_t[p] = _ti_m[p] / ti.max(J * _ti_v0[p], 1e-12)
        # Right Cauchy-Green C = F^T F  (frame-invariant strain object)
        _ti_C_cg[p] = F.transpose() @ F
        # Cauchy stress T = (1/J) P F^T from the active constitutive law
        Fe  = F @ _ti_Fp[p].inverse()
        mu  = _ti_mu[p] * mu_scale * (1.0 - _ti_damage[p])
        lam = _ti_lam[p] * (1.0 - _ti_damage[p])
        P   = ti.Matrix.zero(ti.f32, 3, 3)
        p_model = _ti_model[p]
        if p_model == 0:
            P = _pk1_corotated(Fe, mu, lam)
        elif p_model == 2:
            P = _pk1_sand(Fe, mu, lam)
        elif p_model == 3:
            P = _pk1_stvk(F, mu, lam)
        else:
            kp = kappa
            if _ti_kappa_p[p] > 0.0:
                kp = _ti_kappa_p[p]
            P = _pk1_fluid(Fe, kp, mu)
        _ti_T[p] = (1.0 / ti.max(J, 1e-6)) * P @ F.transpose()


@ti.kernel
def _ti_damage_update(start: ti.i32, end: ti.i32,
                      yield_strain: ti.f32, rate: ti.f32, dt: ti.f32):
    """Grow per-particle damage when local strain exceeds yield_strain.
    Damage is PERMANENT  -  ratchet only, never decreases.  Strain measure
    is (max principal stretch − 1), positive only.  At full damage (d=1)
    the particle's elastic modulus is zero  -  fully softened.

    Only grows when tear mode is active (_ti_tear_active == 1) so that
    normal drops/bounces stay fully elastic.  Press T to start tearing."""
    if _ti_tear_active[None] == 1:
        for p in range(start, end):
            U, sigma, V = ti.svd(_ti_F[p])
            max_stretch = ti.max(sigma[0, 0], ti.max(sigma[1, 1], sigma[2, 2]))
            excess = ti.max(0.0, (max_stretch - 1.0) - yield_strain)
            if excess > 0.0:
                d_new = _ti_damage[p] + rate * excess * dt
                if d_new > 1.0:
                    d_new = 1.0
                _ti_damage[p] = d_new


# Bond-break kernel lives in solver_tl.py because it needs _ti_wnb_tl
# (the WLS neighbour table, owned by the TL solver).


@ti.kernel
def _ti_phase_update(start: ti.i32, end: ti.i32,
                     melt_temp: ti.f32, latent_heat: ti.f32,
                     molten_kappa: ti.f32, heat_capacity: ti.f32,
                     molten_mu: ti.f32):
    """Latent-heat + melting (Stomakhin 2014, Sec 5.9), one-way coupling.

    For a SOLID particle whose temperature exceeds the melt point, the excess
    is banked into the latent-heat buffer U (temperature held at melt_temp  - 
    the latent plateau).  When U fills to L, the particle MELTS:
      - phase → fluid, constitutive law → fluid (_ti_model = 1)
      - shear modulus μ → molten_mu (small residual; 0 = perfectly inviscid
        and spreads to a flat pancake; >0 gives the melt viscous resistance so
        it holds a rounded dome under gravity)
      - per-particle bulk modulus → molten_kappa (resists volume change)
      - deviatoric part of F cleared: F ← Jᴇ^(1/3)·I (fluid is shear-plastic)
      - Fp reset to identity
    Melting only (no re-freeze in v1)."""
    eye = ti.Matrix.identity(ti.f32, 3)
    for p in range(start, end):
        if _ti_phase[p] == 0:                       # solid
            T = _ti_temp[p]
            if T > melt_temp:
                excess = T - melt_temp
                _ti_heat_U[p] = _ti_heat_U[p] + heat_capacity * excess
                _ti_temp[p] = melt_temp              # pin at melt point (latent plateau)
                if _ti_heat_U[p] >= latent_heat:
                    # ---- MELT ----
                    _ti_phase[p]   = 1
                    _ti_model[p]   = 1               # fluid constitutive law
                    _ti_mu[p]      = molten_mu       # residual shear (0=inviscid pancake)
                    _ti_kappa_p[p] = molten_kappa    # fluid pressure stiffness
                    # clear deviatoric component of F → pure dilational
                    Je = _ti_F[p].determinant()
                    s  = ti.pow(ti.max(Je, 1e-6), 1.0 / 3.0)
                    _ti_F[p]  = eye * s
                    _ti_Fp[p] = eye


@ti.kernel
def _ti_thermo_soften(start: ti.i32, end: ti.i32,
                      melt_temp: ti.f32, softening_range: ti.f32):
    """Continuous thermal softening of shear modulus μ(T) for solid particles.

    Linear ramp from μ₀ at T_soft = melt_temp − softening_range down to 0 at
    melt_temp.  Writes _ti_mu[p] every frame so the mechanical substeps see a
    continuously varying stiffness, not a sudden step at phase transition.

    Only acts on solid particles (phase == 0).  Melted particles (phase == 1)
    already have μ = 0 set by _ti_phase_update and are not touched here."""
    for p in range(start, end):
        if _ti_phase[p] == 0:
            T      = _ti_temp[p]
            mu0    = _ti_mu0[p]
            T_soft = melt_temp - softening_range
            if T <= T_soft:
                _ti_mu[p] = mu0
            elif T < melt_temp:
                frac = (T - T_soft) / softening_range
                _ti_mu[p] = mu0 * (1.0 - frac)
            else:
                _ti_mu[p] = 0.0


@ti.kernel
def _ti_molten_floor_spread(start: ti.i32, end: ti.i32,
                            floor_y: ti.f32, dt: ti.f32, strength: ti.f32):
    """Visual floor-spread for molten particles (--demo-spread only).

    NOT physically derived.  Gate with DEMO_SPREAD_ENABLED; default OFF so
    the poster demo uses only MPM grid dynamics for spreading."""
    for p in range(start, end):
        if _ti_phase[p] == 1:
            pos = _ti_pos[p]
            if pos[1] <= floor_y + 0.65:
                r = ti.Vector([pos[0], 0.0, pos[2]])
                rn = r.norm()
                if rn > 1e-5:
                    radial = r / rn
                    proximity = ti.max(0.0, 1.0 - (pos[1] - floor_y) / 0.65)
                    creep = strength * proximity * dt
                    _ti_pos[p][0] += 0.025 * creep * radial[0]
                    _ti_pos[p][2] += 0.025 * creep * radial[2]
                    _ti_vel[p][0] += creep * radial[0]
                    _ti_vel[p][2] += creep * radial[2]
                    _ti_vel[p][0] *= 0.995
                    _ti_vel[p][2] *= 0.995
            if pos[1] > floor_y + 0.12:
                _ti_pos[p][1] = ti.max(floor_y + 0.12, pos[1] - 0.055 * strength * dt)


# ============================================================
# Tear-mode anchors (PhysGaussian-style kinematic pull, for fracture demo)
# ============================================================
_ti_tear_active = ti.field(ti.i32, shape=())   # 0 = off, 1 = on
_ti_tear_active[None] = 0
TEAR_SPEED        = 0.6     # m/s  -  speed of each anchor slab
TEAR_SLAB_FRAC    = 0.20    # top 20% of rest-y becomes "top slab", bottom 20% likewise


@ti.kernel
def _ti_tear_anchor(start: ti.i32, end: ti.i32,
                    rest_y_lo: ti.f32, rest_y_hi: ti.f32,
                    top_thr: ti.f32, bot_thr: ti.f32, dt: ti.f32):
    """When _ti_tear_active is set, override the velocity of particles whose
    rest-y is in the top slab (rest_y > top_thr) or bottom slab (rest_y < bot_thr).
    Top pulls +y, bottom pulls -y.  Middle particles are unaffected and stretch
    between the two anchors."""
    if _ti_tear_active[None] == 1:
        for p in range(start, end):
            ry = _ti_rest[p][1]
            if ry > top_thr:
                _ti_vel[p] = ti.Vector([0.0, TEAR_SPEED, 0.0])
                _ti_pos[p] = _ti_pos[p] + _ti_vel[p] * dt
            elif ry < bot_thr:
                _ti_vel[p] = ti.Vector([0.0, -TEAR_SPEED, 0.0])
                _ti_pos[p] = _ti_pos[p] + _ti_vel[p] * dt


@ti.kernel
def _ti_update_covs():
    for p in range(N_PARTICLES):
        _ti_covs[p] = _ti_F[p] @ _ti_rcov[p] @ _ti_F[p].transpose()


@ti.kernel
def _ti_rigidify_F(start: ti.i32, end: ti.i32):
    """For rigid materials: strip the stretch part of F, keep only rotation.
    Sliced so each entity rigidifies only its own particles."""
    eye = ti.Matrix.identity(ti.f32, 3)
    for p in range(start, end):
        _ti_F[p] = _ti_R[p]
        _ti_U[p] = eye


@ti.kernel
def _ti_shape_gather_com(start: ti.i32, end: ti.i32):
    """GPU pass 1: atomically sum particles [start, end) into _ti_shape_com.
    The scalar accumulator is reused across calls  -  caller divides by (end-start)."""
    _ti_shape_com[None] = ti.Vector([0.0, 0.0, 0.0])
    for p in range(start, end):
        _ti_shape_com[None] += _ti_pos[p]


@ti.kernel
def _ti_shape_gather_H(start: ti.i32, end: ti.i32):
    """GPU pass 2: accumulate H = Σ rest_p ⊗ (pos_p − com) over [start, end)."""
    _ti_shape_buf[None] = ti.Matrix.zero(ti.f32, 3, 3)
    for p in range(start, end):
        d = _ti_pos[p] - _ti_shape_com[None]
        _ti_shape_buf[None] += _ti_rest[p].outer_product(d)


@ti.kernel
def _ti_shape_apply(start: ti.i32, end: ti.i32, alpha: ti.f32):
    """Soft shape apply over [start, end)  -  per-particle floor clamp."""
    com = _ti_shape_com[None]
    R   = _ti_shape_R[None]
    for p in range(start, end):
        _ti_pos[p] += alpha * (R @ _ti_rest[p] + com - _ti_pos[p])
        if _ti_pos[p][1] < CONTACT_ZONE:
            _ti_pos[p][1] = CONTACT_ZONE
            if _ti_vel[p][1] < 0.0:
                _ti_vel[p][1] = 0.0


_ti_body_min_y = ti.field(ti.f32, shape=())


@ti.kernel
def _ti_shape_apply_rigid(start: ti.i32, end: ti.i32, alpha: ti.f32):
    """Rigid shape apply over [start, end): snap to Procrustes target, then
    lift entire entity as a unit until its lowest particle is at CONTACT_ZONE."""
    com = _ti_shape_com[None]
    R   = _ti_shape_R[None]
    for p in range(start, end):
        _ti_pos[p] += alpha * (R @ _ti_rest[p] + com - _ti_pos[p])
    _ti_body_min_y[None] = 1e10
    for p in range(start, end):
        ti.atomic_min(_ti_body_min_y[None], _ti_pos[p][1])
    for p in range(start, end):
        shift = ti.max(0.0, CONTACT_ZONE - _ti_body_min_y[None])
        _ti_pos[p][1] += shift


@ti.kernel
def _ti_zero_C(start: ti.i32, end: ti.i32):
    """Zero APIC C over [start, end)  -  used after a rigid snap to prevent
    stale velocity gradients from injecting lateral momentum next substep."""
    for p in range(start, end):
        _ti_C[p] = ti.Matrix.zero(ti.f32, 3, 3)


@ti.kernel
def _ti_skin_update(
    n_verts: ti.i32,
    idxs:    ti.types.ndarray(dtype=ti.i32, ndim=2),
    weights: ti.types.ndarray(dtype=ti.f32, ndim=2),
    offsets: ti.types.ndarray(dtype=ti.f32, ndim=2),
    out:     ti.types.ndarray(dtype=ti.f32, ndim=2),
):
    """GPU skinning: weighted particle positions + U-stretched rest offset → vertex positions."""
    for v in range(n_verts):
        recon = ti.Vector([0.0, 0.0, 0.0])
        U_loc = ti.Matrix.zero(ti.f32, 3, 3)
        for k in range(K_BIND):
            p      = idxs[v, k]
            w      = weights[v, k]
            recon += w * _ti_pos[p]
            U_loc += w * _ti_U[p]
        x = U_loc @ ti.Vector([offsets[v, 0], offsets[v, 1], offsets[v, 2]]) + recon
        out[v, 0] = x[0]; out[v, 1] = x[1]; out[v, 2] = x[2]



@ti.kernel
def _ti_plasticity_update(mu_scale: ti.f32, dt: ti.f32):
    """
    Multiplicative plasticity:  F = Fe * Fp
    1. Compute Fe = F * Fp^{-1}
    2. Trial PK1 from Fe (μ may include strain-rate stiffening)
    3. Yield: Frobenius ‖P‖ (metals) or von Mises √((3/2)‖dev P‖²) when plastic_j2
    4. Radial return on Fe toward I; optional cap on max principal stretch (tension)
    5. Fp → I with relax_time τ (viscoelasticity)
    """
    eye = ti.Matrix.identity(ti.f32, 3)
    for p in range(N_PARTICLES):
        Fp  = _ti_Fp[p]
        F   = _ti_F[p]
        Fe  = F @ Fp.inverse()
        ys  = _ti_yield[p]
        mu_eff = _mu_eff_sr(p, mu_scale)
        P   = _pk1_corotated(Fe, mu_eff, _ti_lam[p])
        p_norm = P.norm()

        trP = P[0, 0] + P[1, 1] + P[2, 2]
        devP = P - (trP / 3.0) * eye
        q_j2 = ti.sqrt(1.5) * devP.norm()
        use_j2 = _ti_plastic_j2[p]
        eff_q = p_norm
        if use_j2 != 0:
            eff_q = q_j2

        if use_j2 == 2:
            # ---- Drucker-Prager return map (sand/clay) ----
            # Works in Hencky (log-strain) space.  Yield surface is a friction
            # cone parameterised by α = sin(φ)/(√3·cos(φ)) where φ is the
            # friction angle.  Matches PhysGaussian sand_return_mapping
            # (mpm_utils.py:228).
            U_dp, sig_dp, V_dp = ti.svd(Fe)
            if (U_dp @ V_dp.transpose()).determinant() < 0:
                V_dp[0, 2] = -V_dp[0, 2]
                V_dp[1, 2] = -V_dp[1, 2]
                V_dp[2, 2] = -V_dp[2, 2]
                sig_dp[2, 2] = -sig_dp[2, 2]
            sd0 = ti.max(ti.abs(sig_dp[0, 0]), 1e-14)
            sd1 = ti.max(ti.abs(sig_dp[1, 1]), 1e-14)
            sd2 = ti.max(ti.abs(sig_dp[2, 2]), 1e-14)
            eps = ti.Vector([ti.log(sd0), ti.log(sd1), ti.log(sd2)])
            tr_e = eps[0] + eps[1] + eps[2]
            eps_hat = eps - ti.Vector([tr_e / 3.0, tr_e / 3.0, tr_e / 3.0])
            eps_hat_norm = eps_hat.norm() + 1e-8
            dp_alpha = _ti_dp_alpha[p]
            mu_p = mu_eff
            lam_p = _ti_lam[p]
            delta_gamma = (
                eps_hat_norm
                + (3.0 * lam_p + 2.0 * mu_p) / (2.0 * mu_p) * tr_e * dp_alpha
            )
            if delta_gamma > 0.0:
                Fe_new = Fe  # default: no change (overwritten below)
                if tr_e > 0.0:
                    Fe_new = U_dp @ V_dp.transpose()
                else:
                    H = eps - eps_hat * (delta_gamma / eps_hat_norm)
                    s_new = ti.Vector([ti.exp(H[0]), ti.exp(H[1]), ti.exp(H[2])])
                    Sc_new = ti.Matrix([[s_new[0], 0.0, 0.0],
                                        [0.0, s_new[1], 0.0],
                                        [0.0, 0.0, s_new[2]]])
                    Fe_new = U_dp @ Sc_new @ V_dp.transpose()
                _ti_Fp[p] = Fe_new.inverse() @ F
        elif use_j2 == 3:
            # ---- Log-strain J2 return map (StVK metal, PhysGaussian-style) ----
            # Works in Hencky (log-strain) space using SVD of F (not Fe  -  this
            # path does NOT use Fp; it overwrites F directly with F_elastic,
            # matching PhysGaussian's von_mises_return_mapping at mpm_utils.py:78.
            # Hardening: yield_stress grows with plastic work via xi parameter.
            U_m, sig_m, V_m = ti.svd(F)
            if (U_m @ V_m.transpose()).determinant() < 0:
                V_m[0, 2] = -V_m[0, 2]
                V_m[1, 2] = -V_m[1, 2]
                V_m[2, 2] = -V_m[2, 2]
                sig_m[2, 2] = -sig_m[2, 2]
            sm0 = ti.max(ti.abs(sig_m[0, 0]), 1e-4)
            sm1 = ti.max(ti.abs(sig_m[1, 1]), 1e-4)
            sm2 = ti.max(ti.abs(sig_m[2, 2]), 1e-4)
            eps_m = ti.Vector([ti.log(sm0), ti.log(sm1), ti.log(sm2)])
            tr_em = eps_m[0] + eps_m[1] + eps_m[2]
            # Trial Kirchhoff stress in principal space: τ = 2μ·ε + λ·tr(ε)·I
            tau_v = 2.0 * mu_eff * eps_m + _ti_lam[p] * tr_em * ti.Vector([1.0, 1.0, 1.0])
            tau_tr = tau_v[0] + tau_v[1] + tau_v[2]
            # Deviatoric part  -  check yield (J2: |dev τ| > yield)
            cond = tau_v - ti.Vector([tau_tr / 3.0, tau_tr / 3.0, tau_tr / 3.0])
            cond_norm = cond.norm()
            if cond_norm > ys and mu_eff > 1e-6:
                # Deviatoric strain
                eps_hat = eps_m - ti.Vector([tr_em / 3.0, tr_em / 3.0, tr_em / 3.0])
                eps_hat_n = eps_hat.norm() + 1e-8
                # Plastic multiplier in log-strain space
                delta_gamma_m = eps_hat_n - ys / (2.0 * mu_eff)
                # Project onto yield surface (tangential return in log-strain)
                eps_new = eps_m - (delta_gamma_m / eps_hat_n) * eps_hat
                sig_new = ti.Vector([ti.exp(eps_new[0]),
                                     ti.exp(eps_new[1]),
                                     ti.exp(eps_new[2])])
                Sc_new = ti.Matrix([[sig_new[0], 0.0, 0.0],
                                    [0.0, sig_new[1], 0.0],
                                    [0.0, 0.0, sig_new[2]]])
                # F_elastic OVERWRITES F (no separate Fp for this path).
                # This matches PhysGaussian: F is the elastic part after return.
                _ti_F[p] = U_m @ Sc_new @ V_m.transpose()
                # Reset Fp to identity  -  the StVK path doesn't carry plastic
                # state in Fp; all plastic memory is "absorbed" into the
                # overwrite of F itself.
                _ti_Fp[p] = eye
                # Hardening: grow yield stress proportional to plastic work
                xi = _ti_hardening_xi[p]
                if xi > 1e-6 and delta_gamma_m > 0.0:
                    _ti_yield[p] = ys + 2.0 * mu_eff * xi * delta_gamma_m
        elif ys < 1e8 and eff_q > ys:
            # ---- J2 / Frobenius radial return (F-space approximation) ----
            scale     = ys / (eff_q + 1e-8)
            Fe_scaled = scale * Fe + (1.0 - scale) * eye
            _ti_Fp[p] = Fe_scaled.inverse() @ F

        # Viscoelastic relaxation  -  Fp → I with exp decay over relax_time τ
        tau = _ti_relax[p]
        if tau > 1e-6:
            decay = ti.exp(-dt / tau)
            _ti_Fp[p] = eye + (_ti_Fp[p] - eye) * decay

        # Rankine-style tension cap on elastic principal stretches (σ_i ≤ 1 + cap)
        cap = _ti_tension_cap[p]
        if cap > 1e-8:
            Fp2 = _ti_Fp[p]
            Fe2 = F @ Fp2.inverse()
            U, sig, V = ti.svd(Fe2)
            s0 = ti.min(sig[0, 0], 1.0 + cap)
            s1 = ti.min(sig[1, 1], 1.0 + cap)
            s2 = ti.min(sig[2, 2], 1.0 + cap)
            Sc = ti.Matrix([[s0, 0.0, 0.0],
                            [0.0, s1, 0.0],
                            [0.0, 0.0, s2]])
            Fe_cap = U @ Sc @ V.transpose()
            _ti_Fp[p] = Fe_cap.inverse() @ F

        # ---- Fp singular-value clamp (standard MPM plasticity safety) ----
        # Without this, plastic strain accumulates unboundedly under repeated
        # yield events (the radial return above absorbs strain into Fp each
        # frame; relaxation alone may not catch up if kicks come faster than
        # the relax_time).  Clamping σ(Fp) ∈ [0.3, 3.0] keeps Fe = F·Fp⁻¹
        # bounded so the next-frame trial stress can't explode.  This is
        # Stomakhin-style bookkeeping  -  every reference MPM plasticity has it.
        # Drucker-Prager materials do not use yield_stress as their yield
        # predicate, so include them explicitly.
        # StVK metal path (use_j2 == 3) holds Fp at identity by construction,
        # so the clamp is unnecessary and would force a no-op SVD.
        if (ys < 1e8 or use_j2 == 2) and use_j2 != 3:
            Up, sp, Vp = ti.svd(_ti_Fp[p])
            sp0 = ti.min(ti.max(sp[0, 0], 0.5), 2.0)
            sp1 = ti.min(ti.max(sp[1, 1], 0.5), 2.0)
            sp2 = ti.min(ti.max(sp[2, 2], 0.5), 2.0)
            Scp = ti.Matrix([[sp0, 0.0, 0.0],
                             [0.0, sp1, 0.0],
                             [0.0, 0.0, sp2]])
            _ti_Fp[p] = Up @ Scp @ Vp.transpose()


@ti.kernel
def _ti_diagnostics(kappa: ti.f32, mu_scale: ti.f32, model: ti.i32):
    for p in range(N_PARTICLES):
        F  = _ti_F[p]
        mu = _ti_mu[p] * mu_scale
        P  = ti.Matrix.zero(ti.f32, 3, 3)
        p_model = _ti_model[p]
        if p_model == 0:
            P = _pk1_corotated(F, mu, _ti_lam[p])
        elif p_model == 2:
            P = _pk1_sand(F, mu, _ti_lam[p])
        elif p_model == 3:
            P = _pk1_stvk(F, mu, _ti_lam[p])
        else:
            kp = kappa
            if _ti_kappa_p[p] > 0.0:
                kp = _ti_kappa_p[p]
            P = _pk1_fluid(F, kp, mu)
        _ti_sn[p] = P.norm()
        _, sigma, _ = ti.svd(F)
        _ti_sm[p] = ti.min(ti.min(sigma[0, 0], sigma[1, 1]), sigma[2, 2])


# ---- PBF spatial hash helpers -------------------------------------------

@ti.func
def _pbf_cell_flat(ix: ti.i32, iy: ti.i32, iz: ti.i32) -> ti.i32:
    return ix * _PBF_GRID_N * _PBF_GRID_N + iy * _PBF_GRID_N + iz


@ti.kernel
def _ti_pbf_build_hash():
    """Clear then populate the spatial hash from current predicted positions.
    Two sequential top-level loops: clear runs to completion before insert."""
    for ci in range(_PBF_TOTAL_CELLS):
        _pbf_cell_count[ci] = 0
    for p in range(N_PARTICLES):
        pos  = _ti_pred[p]
        ix   = ti.cast(ti.floor((pos[0] - _PBF_GRID_ORIGIN) / _PBF_H), ti.i32)
        iy   = ti.cast(ti.floor((pos[1] - _PBF_GRID_ORIGIN) / _PBF_H), ti.i32)
        iz   = ti.cast(ti.floor((pos[2] - _PBF_GRID_ORIGIN) / _PBF_H), ti.i32)
        ix   = ti.max(0, ti.min(ix, _PBF_GRID_N - 1))
        iy   = ti.max(0, ti.min(iy, _PBF_GRID_N - 1))
        iz   = ti.max(0, ti.min(iz, _PBF_GRID_N - 1))
        cell = _pbf_cell_flat(ix, iy, iz)
        slot = ti.atomic_add(_pbf_cell_count[cell], 1)
        if slot < _PBF_MAX_PER_CELL:
            _pbf_cell_plist[cell, slot] = p


# ---- PBF kernels (fluid only) -------------------------------------------
# Position-Based Fluids: Macklin & Müller 2013, XPBD variant from paper.
# Spatial hash O(N·k) where k≈27 cells × ~6 particles  -  one build per substep.
# Per substep: predict → build_hash → [lambda + dx + apply] × ITER → finalize → viscosity.

@ti.kernel
def _ti_pbf_predict(dt: ti.f32):
    """Predict positions under gravity; clamp at floor."""
    for i in range(N_PARTICLES):
        v_new      = _ti_vel[i] + dt * ti.Vector([0.0, -9.8, 0.0])
        pred       = _ti_pos[i] + dt * v_new
        if pred[1] < CONTACT_ZONE:
            pred[1] = CONTACT_ZONE
        _ti_pred[i] = pred


@ti.kernel
def _ti_pbf_lambda(rho0: ti.f32):
    """Compute per-particle density ρᵢ and Lagrange multiplier λᵢ.

    λᵢ = −Cᵢ / (Σₖ |∇_pk Cᵢ|²/m + ε)
    where Cᵢ = ρᵢ/ρ₀ − 1  (incompressibility constraint).
    """
    m = PARTICLE_MASS
    for i in range(N_PARTICLES):
        xi      = _ti_pred[i]
        rho     = m * _pbf_W(0.0)   # self-contribution (r=0)
        grad_i  = ti.Vector([0.0, 0.0, 0.0])   # ∇_xi Cᵢ  (accumulated)
        grad_sq = 0.0                            # Σⱼ |∇_xj Cᵢ|²

        cix = ti.cast(ti.floor((xi[0] - _PBF_GRID_ORIGIN) / _PBF_H), ti.i32)
        ciy = ti.cast(ti.floor((xi[1] - _PBF_GRID_ORIGIN) / _PBF_H), ti.i32)
        ciz = ti.cast(ti.floor((xi[2] - _PBF_GRID_ORIGIN) / _PBF_H), ti.i32)
        for ddx in range(-1, 2):
            for ddy in range(-1, 2):
                for ddz in range(-1, 2):
                    nx = cix + ddx; ny = ciy + ddy; nz = ciz + ddz
                    if nx >= 0 and nx < _PBF_GRID_N and ny >= 0 and ny < _PBF_GRID_N and nz >= 0 and nz < _PBF_GRID_N:
                        cell = _pbf_cell_flat(nx, ny, nz)
                        cnt  = ti.min(_pbf_cell_count[cell], _PBF_MAX_PER_CELL)
                        for k in range(cnt):
                            j   = _pbf_cell_plist[cell, k]
                            if j != i:
                                xij = xi - _ti_pred[j]
                                r2  = xij.norm_sqr()
                                if r2 < _PBF_H2:
                                    r    = ti.sqrt(r2)
                                    rho += m * _pbf_W(r)
                                    gw   = (m / rho0) * _pbf_gradW(xij, r)
                                    grad_i  += gw
                                    grad_sq += gw.norm_sqr()

        _ti_rho[i] = rho
        C_i        = rho / rho0 - 1.0
        # denom = (1/m)(|∇_xi C|² + Σⱼ|∇_xj C|²) + ε
        denom      = (grad_i.norm_sqr() + grad_sq) / m + _PBF_EPS
        _ti_lambda[i] = -C_i / denom


@ti.kernel
def _ti_pbf_compute_dx(rho0: ti.f32):
    """Compute position correction Δxᵢ = (1/ρ₀) Σⱼ (λᵢ+λⱼ) ∇W(xᵢ−xⱼ).

    Also adds distance constraint correction for particle separation.
    Written to _ti_dx; not applied yet (avoids race conditions).
    """
    m = PARTICLE_MASS
    for i in range(N_PARTICLES):
        xi = _ti_pred[i]
        dx = ti.Vector([0.0, 0.0, 0.0])

        cix = ti.cast(ti.floor((xi[0] - _PBF_GRID_ORIGIN) / _PBF_H), ti.i32)
        ciy = ti.cast(ti.floor((xi[1] - _PBF_GRID_ORIGIN) / _PBF_H), ti.i32)
        ciz = ti.cast(ti.floor((xi[2] - _PBF_GRID_ORIGIN) / _PBF_H), ti.i32)
        for ddx in range(-1, 2):
            for ddy in range(-1, 2):
                for ddz in range(-1, 2):
                    nx = cix + ddx; ny = ciy + ddy; nz = ciz + ddz
                    if nx >= 0 and nx < _PBF_GRID_N and ny >= 0 and ny < _PBF_GRID_N and nz >= 0 and nz < _PBF_GRID_N:
                        cell = _pbf_cell_flat(nx, ny, nz)
                        cnt  = ti.min(_pbf_cell_count[cell], _PBF_MAX_PER_CELL)
                        for k in range(cnt):
                            j   = _pbf_cell_plist[cell, k]
                            if j != i:
                                xij = xi - _ti_pred[j]
                                r2  = xij.norm_sqr()
                                if r2 < _PBF_H2:
                                    r   = ti.sqrt(r2)
                                    dx += (_ti_lambda[i] + _ti_lambda[j]) * _pbf_gradW(xij, r)
                                    if r < _PBF_D0 and r > 1e-6:
                                        dx += 0.5 * (_PBF_D0 - r) / r * xij

        _ti_dx[i] = dx / rho0


@ti.kernel
def _ti_pbf_apply_dx():
    """Apply position corrections and re-clamp floor."""
    for i in range(N_PARTICLES):
        p = _ti_pred[i] + _ti_dx[i]
        if p[1] < CONTACT_ZONE:
            p[1] = CONTACT_ZONE
        _ti_pred[i] = p


@ti.kernel
def _ti_pbf_finalize(dt: ti.f32, restitution: ti.f32):
    """Update velocity from position change; damp floor-bounce by restitution."""
    for i in range(N_PARTICLES):
        new_v = (_ti_pred[i] - _ti_pos[i]) / dt
        # Floor: if particle was corrected upward by floor, apply restitution
        if _ti_pos[i][1] <= CONTACT_ZONE + 1e-4 and new_v[1] > 0.0:
            new_v[1] *= restitution
        spd = new_v.norm()
        if spd > MAX_SPEED:
            new_v = new_v * (MAX_SPEED / spd)
        _ti_vel[i] = new_v
        _ti_pos[i] = _ti_pred[i]


@ti.kernel
def _ti_pbf_viscosity(damping: ti.f32):
    """XSPH viscosity: v'ᵢ += c·Σⱼ (vⱼ−vᵢ)·W(|xᵢ−xⱼ|) / ρⱼ
    Smooths the velocity field  -  prevents particle clumping artifacts.
    Also applies linear damping (air drag)."""
    for i in range(N_PARTICLES):
        xi   = _ti_pos[i]
        vi   = _ti_vel[i]
        dv   = ti.Vector([0.0, 0.0, 0.0])
        cix = ti.cast(ti.floor((xi[0] - _PBF_GRID_ORIGIN) / _PBF_H), ti.i32)
        ciy = ti.cast(ti.floor((xi[1] - _PBF_GRID_ORIGIN) / _PBF_H), ti.i32)
        ciz = ti.cast(ti.floor((xi[2] - _PBF_GRID_ORIGIN) / _PBF_H), ti.i32)
        for ddx in range(-1, 2):
            for ddy in range(-1, 2):
                for ddz in range(-1, 2):
                    nx = cix + ddx; ny = ciy + ddy; nz = ciz + ddz
                    if nx >= 0 and nx < _PBF_GRID_N and ny >= 0 and ny < _PBF_GRID_N and nz >= 0 and nz < _PBF_GRID_N:
                        cell = _pbf_cell_flat(nx, ny, nz)
                        cnt  = ti.min(_pbf_cell_count[cell], _PBF_MAX_PER_CELL)
                        for k in range(cnt):
                            j    = _pbf_cell_plist[cell, k]
                            if j != i:
                                xij = xi - _ti_pos[j]
                                r2  = xij.norm_sqr()
                                if r2 < _PBF_H2:
                                    r    = ti.sqrt(r2)
                                    rhoj = ti.max(_ti_rho[j], 1e-6)
                                    dv  += (PARTICLE_MASS / rhoj) * (_ti_vel[j] - vi) * _pbf_W(r)
        _ti_vel[i] = (vi + _PBF_XSPH * dv) * (1.0 - damping)


@ti.kernel
def _ti_pbf_sample_density():
    """Compute density at rest positions (used once at reset to find ρ₀)."""
    m = PARTICLE_MASS
    for i in range(N_PARTICLES):
        xi  = _ti_pos[i]
        rho = m * _pbf_W(0.0)
        for j in range(N_PARTICLES):
            if i != j:
                r2 = (xi - _ti_pos[j]).norm_sqr()
                if r2 < _PBF_H2:
                    rho += m * _pbf_W(ti.sqrt(r2))
        _ti_rho[i] = rho


# ============================================================
# Solver modules (late import  -  they reference fields defined above)
# ============================================================
import solver_tl
import evaluate_drift

# solver_ul allocates a world-frame MPM grid sized to MPM_GRID_H, which can
# balloon to millions of cells for fine-resolution custom meshes.  Only load
# it when actually needed (UL-only single-entity runs or the compare scene).
solver_ul = None
if SOLVER == "ul_mlsmpm" or SCENE in ("compare", "collide"):
    import solver_ul
    # Floor BC (PhysGaussian add_bounding_box / add_surface_collider pattern):
    # zero downward grid velocity at nodes in the floor contact band.  No mass
    # to tune, unconditionally stable (it's a velocity constraint, not a force).
    # init_floor() is still available for "floor-as-body" experiments but not
    # used in the standard PhysGaussian-equivalent path.

# Solver dispatch table  -  clean toggle, no scattered if-statements
_SOLVER_MODULES = {
    "tl_apic":   solver_tl,
    "ul_mlsmpm": solver_ul,   # None when not loaded; not indexed unless requested
}


def fracture_compute_components(start: int, end: int) -> np.ndarray:
    """Return per-particle component IDs for the entity slice [start, end).
    Two particles are in the same component iff there is a path of INTACT
    WLS bonds between them.  Uses the WLS neighbour table from solver_tl;
    so this only makes sense for TL particles (UL has no bonds)."""
    N = end - start
    intact = _ti_bond_intact.to_numpy()[start:end]   # (N, K_PHYS) int32
    nbrs   = solver_tl._ti_wnb_tl.to_numpy()[start:end] - start  # local-slice indices
    # Build sparse adjacency (intact bonds only; ignore self / out-of-slice neighbours)
    rows, cols = [], []
    for p in range(N):
        for k in range(K_PHYS):
            if intact[p, k] == 1:
                nb = nbrs[p, k]
                if 0 <= nb < N and nb != p:
                    rows.append(p); cols.append(nb)
    if not rows:
        # Pathological: no intact bonds anywhere → every particle its own component
        return np.arange(N, dtype=np.int32)
    data = np.ones(len(rows), dtype=np.int32)
    adj  = csr_matrix((data, (rows, cols)), shape=(N, N))
    _, labels = connected_components(adj, directed=False)
    return labels.astype(np.int32)


def fracture_rebind_skin(orig_idxs:   np.ndarray,
                         orig_weights: np.ndarray,
                         components:   np.ndarray) -> np.ndarray:
    """Given the original IDW binding (idxs/weights per vertex) and the current
    component label per particle, return a new weights array in which each
    vertex's weights are zeroed for particles outside the vertex's dominant
    component, then renormalised.  Mesh visually splits where the particle
    graph splits.

    orig_idxs:    (V, K) int  -  original k-nearest particle indices per vertex
    orig_weights: (V, K) f32  -  original IDW weights (sum-to-1 per vertex)
    components:   (N,)   int  -  connected-component label per particle
    returns:      (V, K) f32  -  masked + renormalised weights
    """
    vertex_comp = components[orig_idxs]               # (V, K)  -  comp of each bound particle
    # Find each vertex's dominant component (majority by IDW weight, not count)
    V, K = orig_idxs.shape
    # Build (V, max_comp+1) one-hot accumulation weighted by IDW
    max_c = int(components.max()) + 1
    w_per_comp = np.zeros((V, max_c), dtype=np.float32)
    for k in range(K):
        np.add.at(w_per_comp, (np.arange(V), vertex_comp[:, k]), orig_weights[:, k])
    dominant = w_per_comp.argmax(axis=1)               # (V,)  -  dominant component per vertex
    # Mask: keep weight only where the bound particle is in vertex's dominant component
    keep_mask = (vertex_comp == dominant[:, None])     # (V, K) bool
    new_w = orig_weights * keep_mask.astype(np.float32)
    # Renormalise so the remaining weights still sum to 1 per vertex; if a vertex
    # has zero remaining weight (all bound particles disappeared from its dominant
    # component for some reason), fall back to original weights so it doesn't NaN.
    row_sum = new_w.sum(axis=1, keepdims=True)
    safe    = row_sum.squeeze(-1) > 1e-6
    new_w[safe] = new_w[safe] / row_sum[safe]
    new_w[~safe] = orig_weights[~safe]
    return new_w.astype(np.float32)


class GaussianBallSim:
    """
    One entity in the scene.  Owns a contiguous slice [start, end) of the
    shared Taichi particle fields plus a solver assignment (TL-APIC or
    UL-MLS-MPM).  In single-entity mode there's just one instance covering
    [0, N_PARTICLES)  -  identical observable behaviour to the original code.
    In compare mode there are two instances on disjoint slices, each running
    its own solver and writing to its own grid (TL fields in solver_tl, UL
    fields in solver_ul, no cross-talk).

    Per-frame post-processing (polar decomp, plasticity, covariances,
    diagnostics) is run once per entity's step() since each entity needs
    its own diagnostics with its own mu_scale/kappa.  PBF and fluid paths
    remain unchanged.
    """

    def __init__(self,
                 material_key: str = "1",
                 *,
                 entity_id: int = 0,
                 start: int = 0,
                 end: int = None,
                 solver_name: str = None,
                 spawn_position=None,
                 grid_h: float = None):
        self.entity_id     = entity_id
        self.start         = int(start)
        self.end           = int(end) if end is not None else N_PARTICLES
        self.solver_name   = solver_name if solver_name is not None else SOLVER
        self.solver_module = _SOLVER_MODULES[self.solver_name]
        self.spawn_position = (np.asarray(spawn_position, dtype=np.float64)
                               if spawn_position is not None else BALL_START.astype(np.float64))
        self.grid_h        = float(grid_h) if grid_h is not None else float(MPM_GRID_H)
        self.material_key  = material_key
        # mpm_n_grid filled in by _build_neighbourhood for TL; 0 for UL (no rest grid).
        self.mpm_n_grid    = 0
        self._build_neighbourhood()
        self.reset()

    # ---- precompute & material -----------------------------------------

    def _build_neighbourhood(self):
        """Precompute TL stencil + WLS for this entity's rest slice and upload
        into solver_tl's fields at offset self.start.  UL doesn't need a
        rest-frame stencil  -  its world grid is rebuilt every substep  -  but
        rest positions and V0/mass still get uploaded into the shared particle
        struct so shape_restore and UL P2G have what they need."""
        rest_slice = REST_POS[self.start:self.end]
        if self.solver_name == "tl_apic":
            (self.mpm_node_idx, self.mpm_w_ip, self.mpm_dw_ip,
             self.mpm_dpos_ip, self.mpm_n_grid) = build_mpm_grid(rest_slice, grid_h=self.grid_h)
            (self.wls_nbs, self.wls_weights,
             self.wls_B_mats, self.wls_B_invs) = build_F_neighbourhood(rest_slice, k=K_PHYS)
            _ti_upload_stencil(
                self.mpm_node_idx, self.mpm_w_ip, self.mpm_dw_ip, self.mpm_dpos_ip,
                self.wls_nbs, self.wls_weights, self.wls_B_mats, self.wls_B_invs,
                rest_slice, slice_start=self.start,
            )
        else:
            # UL: upload rest_pos, V0, mass into the shared particle struct only.
            N = self.end - self.start
            sl = slice(self.start, self.end)
            cur_rest = _ti_rest.to_numpy(); cur_rest[sl] = rest_slice.astype(np.float32); _ti_rest.from_numpy(cur_rest)
            cur_v0   = _ti_v0.to_numpy()
            # V0 from this entity may be scalar (uniform) or array (grid-fill).
            entity_v0 = _PER_ENTITY_V0[self.entity_id]
            if isinstance(entity_v0, np.ndarray):
                cur_v0[sl] = entity_v0[:N].astype(np.float32)
            else:
                cur_v0[sl] = float(entity_v0)
            _ti_v0.from_numpy(cur_v0)
            cur_m    = _ti_m.to_numpy();    cur_m[sl]    = PARTICLE_MASS;               _ti_m.from_numpy(cur_m)

    def _load_material(self, key):
        """Copy material parameters into instance vars and write per-particle
        params into the shared Taichi fields AT THIS ENTITY'S SLICE."""
        mat                = MATERIALS[key]
        self.material_name = mat["name"]
        self.mat           = mat
        self.damping       = mat["damping"]
        self.restitution   = mat["restitution"]
        self.shape_alpha   = mat["shape_alpha"]
        # Fracture is active if the material opts in via "fracture": True OR if
        # the global --fracture CLI flag is set.
        self.fracture_active = FRACTURE_ENABLED or bool(mat.get("fracture", False))
        self.yield_strain    = float(mat.get("yield_strain", DAMAGE_YIELD_STRAIN))
        self.damage_rate     = float(mat.get("damage_rate",  DAMAGE_RATE))
        s = mat["cov_spread"]
        rest_slice = REST_POS[self.start:self.end]
        # When loaded from a 3DGS PLY, use the real per-particle covariance
        # (anisotropic scale + rotation from the trained model) instead of the
        # procedural disc-shaped rest cov.  This preserves the Gaussian shape
        # signature from the source scene.
        if SHAPE == "ply" and self.entity_id < len(_PLY_COVS_PER_ENTITY):
            self.rest_covs = _PLY_COVS_PER_ENTITY[self.entity_id].astype(np.float32)
        elif (self.entity_id < len(_PER_ENTITY_REST_COVS)
              and _PER_ENTITY_REST_COVS[self.entity_id] is not None):
            # Grid-fill: each particle has an isotropic cov sized to its share
            # of the cell.  Visual ellipsoid radius = (V0·3/4π)^(1/3)  -  perfectly
            # represents the particle's volume regardless of N.
            self.rest_covs = _PER_ENTITY_REST_COVS[self.entity_id].astype(np.float32)
        else:
            self.rest_covs = make_base_covs(rest_slice, s, s)

        r = mat["restitution"]
        ln_r = np.log(np.clip(r, 0.01, 0.99))
        density   = float(mat.get("density", 1.0))
        # Particle mass = ρ · V_p (continuum-correct).  V_p is per-particle
        # under --grid-fill, scalar otherwise; either way m_p reflects the
        # material density.  Old formula `density · (1/N)` made m_p scale as
        # 1/N which broke damping coefficients and penalty floor stability.
        entity_v0 = _PER_ENTITY_V0[self.entity_id]
        if isinstance(entity_v0, np.ndarray):
            self.particle_mass_array = (density * entity_v0).astype(np.float32)
            self.particle_mass = float(self.particle_mass_array.mean())
        else:
            self.particle_mass_array = None
            self.particle_mass = density * float(entity_v0)
        # Gripper still uses the old penalty spring (FLOOR_K) for kinematic
        # plate contact.  floor_damp tunes its damping ratio for restitution=R.
        self.floor_damp = (-2.0 * np.sqrt(FLOOR_K * self.particle_mass) * ln_r
                           / np.sqrt(np.pi**2 + ln_r**2))
        # Coulomb friction coefficient  -  μ·|N| cap on tangent force.
        self.friction = float(mat.get("friction", 0.5))

        N = self.end - self.start
        sl = slice(self.start, self.end)
        def _set_slice(field, val):
            cur = field.to_numpy()
            cur[sl] = val
            field.from_numpy(cur)
        # m_p = ρ · V_p  -  per-particle when grid-fill is active (variable cells),
        # uniform fallback otherwise.  Continuum-correct: m_p depends on material
        # density and the particle's owned volume, NOT on the particle count.
        if self.particle_mass_array is not None:
            _set_slice(_ti_m, self.particle_mass_array)
        else:
            _set_slice(_ti_m, np.full(N, self.particle_mass, dtype=np.float32))
        _set_slice(_ti_mu,      np.full(N, float(mat["mu"]),    dtype=np.float32))
        _set_slice(_ti_mu0,     np.full(N, float(mat["mu"]),    dtype=np.float32))
        _set_slice(_ti_lam,     np.full(N, float(mat["lam"]),   dtype=np.float32))
        alpha = float(mat["damping"]) / self.particle_mass
        _set_slice(_ti_damping, np.full(N, alpha,                              dtype=np.float32))
        _set_slice(_ti_yield,   np.full(N, float(mat.get("yield_stress", 1e9)), dtype=np.float32))
        _set_slice(_ti_relax,   np.full(N, float(mat.get("relax_time",   0.0)), dtype=np.float32))
        # plastic_j2 values: 0=Frobenius, 1=J2 (von Mises), 2=Drucker-Prager
        j2_val = mat.get("plastic_j2", 0)
        j2 = int(j2_val) if isinstance(j2_val, (int, float)) else (1 if j2_val else 0)
        _set_slice(_ti_plastic_j2, np.full(N, j2, dtype=np.int32))
        _set_slice(_ti_tension_cap, np.full(N, float(mat.get("tension_stretch_cap", 0.0)), dtype=np.float32))
        _set_slice(_ti_sr_gamma, np.full(N, float(mat.get("strain_rate_gamma", 0.0)), dtype=np.float32))
        # Drucker-Prager friction cone α = sin(φ)/(√3·cos(φ)).  Non-zero only
        # for model="sand" materials; defaults to 0 (no DP) when friction_angle
        # is absent.
        phi_deg = float(mat.get("friction_angle", 0.0))
        if phi_deg > 0.0:
            phi = np.radians(phi_deg)
            dp_alpha = np.sin(phi) / (np.sqrt(3.0) * np.cos(phi))
        else:
            dp_alpha = 0.0
        _set_slice(_ti_dp_alpha, np.full(N, dp_alpha, dtype=np.float32))
        # Per-particle constitutive law dispatch.
        # 0=corotated, 1=fluid, 2=sand (Drucker-Prager), 3=stvk (metal)
        mat_model_str = mat.get("model", "corotated")
        if mat_model_str == "sand":
            model_int = 2
        elif mat_model_str == "fluid":
            model_int = 1
        elif mat_model_str == "stvk":
            model_int = 3
        else:
            model_int = 0
        _set_slice(_ti_model, np.full(N, model_int, dtype=np.int32))
        # Per-particle hardening xi (PhysGaussian-style work hardening for metals)
        _set_slice(_ti_hardening_xi, np.full(N, float(mat.get("hardening_xi", 0.0)), dtype=np.float32))
        # Thermodynamics state.  Latent heat starts empty for solids and full
        # for materials that are already fluid.  Molten particles later switch
        # _ti_model to fluid and store their pressure stiffness in _ti_kappa_p.
        base_kappa = float(mat.get("kappa", 0.0)) if model_int == 1 else 0.0
        base_phase = 1 if model_int == 1 else 0
        base_U = float(mat.get("latent_heat", 0.0)) if base_phase == 1 else 0.0
        _set_slice(_ti_temp, np.full(N, AMBIENT_TEMP, dtype=np.float32))
        _set_slice(_ti_heat_U, np.full(N, base_U, dtype=np.float32))
        _set_slice(_ti_phase, np.full(N, base_phase, dtype=np.int32))
        _set_slice(_ti_kappa_p, np.full(N, base_kappa, dtype=np.float32))
        cur_rcov = _ti_rcov.to_numpy()
        cur_rcov[sl] = self.rest_covs.astype(np.float32)
        _ti_rcov.from_numpy(cur_rcov)

    # ---- GPU upload (slice-aware) --------------------------------------

    def _upload_to_gpu(self):
        """Push this entity's numpy state into its slice of the shared fields."""
        sl = slice(self.start, self.end)
        for field, arr in [(_ti_pos, self.positions), (_ti_vel, self.velocities),
                           (_ti_F,   self.Fs),         (_ti_Fp,  self.Fp),
                           (_ti_C,   self.Cs)]:
            cur = field.to_numpy()
            cur[sl] = arr.astype(np.float32)
            field.from_numpy(cur)

    # ---- lifecycle ------------------------------------------------------

    def reset(self):
        """Return this entity to its initial state  -  particles at spawn pose,
        F = I, zero velocity, zero APIC C, zero plastic strain."""
        self._load_material(self.material_key)
        N = self.end - self.start
        rest_slice      = REST_POS[self.start:self.end]
        self.positions  = rest_slice + self.spawn_position
        self.velocities = np.zeros((N, 3))
        self.mu_scale   = 1.0
        self.Fs         = np.tile(np.eye(3), (N, 1, 1))
        self.Fp         = np.tile(np.eye(3), (N, 1, 1))
        self.Cs         = np.zeros((N, 3, 3))
        self.Rs         = np.tile(np.eye(3), (N, 1, 1))
        self.Us         = np.tile(np.eye(3), (N, 1, 1))
        self.covs       = self.rest_covs.copy()
        self.dropped    = False
        self.frame      = 0
        self._substep_acc = 0
        self.diag_stress_norm = np.zeros(N)
        self.diag_sigma_min   = np.ones(N)
        self.diag_rho_mean    = 0.0
        self.thermo_molten    = 0
        self.thermo_temp_min  = AMBIENT_TEMP
        self.thermo_temp_max  = AMBIENT_TEMP
        self.thermo_U_mean    = 0.0
        self._upload_to_gpu()
        # Reset fracture state for this entity's slice  -  damage=0, bonds intact.
        sl = slice(self.start, self.end)
        cur_d = _ti_damage.to_numpy(); cur_d[sl] = 0.0; _ti_damage.from_numpy(cur_d)
        cur_b = _ti_bond_intact.to_numpy(); cur_b[sl, :] = 1; _ti_bond_intact.from_numpy(cur_b)
        if self.mat["model"] == "fluid":
            _ti_pbf_sample_density()
            self.pbf_rho0 = float(_ti_rho.to_numpy()[self.start:self.end].mean())
        else:
            self.pbf_rho0 = 1.0

    def apply_material(self, key):
        """Switch material mid-simulation while keeping particle state."""
        self.material_key = key
        sl = slice(self.start, self.end)
        # download current slice
        self.positions  = _ti_pos.to_numpy()[sl].astype(float)
        self.velocities = _ti_vel.to_numpy()[sl].astype(float)
        self.Fs         = _ti_F.to_numpy()[sl].astype(float)
        self.Fp         = _ti_Fp.to_numpy()[sl].astype(float)
        self.Cs         = _ti_C.to_numpy()[sl].astype(float)
        prev = (self.positions.copy(), self.velocities.copy(),
                self.Fs.copy(), self.Fp.copy(), self.Cs.copy(),
                self.dropped, self.frame, self.mu_scale)
        self._load_material(key)
        (self.positions, self.velocities, self.Fs, self.Fp, self.Cs,
         self.dropped, self.frame, self.mu_scale) = prev
        self._upload_to_gpu()
        if self.mat["model"] == "fluid":
            _ti_pbf_sample_density()
            self.pbf_rho0 = float(_ti_rho.to_numpy()[sl].mean())
        else:
            self.pbf_rho0 = 1.0

    def drop(self):
        """Reset this entity's particles to spawn pose and release the simulation.

        When THERMO_ENABLED and the material is meltable, the body is pre-positioned
        so its lowest particle sits just above the floor (CONTACT_ZONE + 0.03) instead
        of falling from spawn height.  This ensures floor heat contact starts on frame 1
        regardless of which run path (run() or run_gs_headless) is used."""
        N = self.end - self.start
        rest_slice      = REST_POS[self.start:self.end]
        self.positions  = rest_slice + self.spawn_position
        self.velocities = np.zeros((N, 3))
        self.Cs         = np.zeros((N, 3, 3))
        self.dropped    = True
        self.frame      = 0          # restart frame counter on every drop
        if THERMO_ENABLED and "melt_temp" in self.mat:
            min_y = float(self.positions[:, 1].min())
            self.positions[:, 1] += (CONTACT_ZONE + 0.03) - min_y
        sl = slice(self.start, self.end)
        for field, arr in [(_ti_pos, self.positions), (_ti_vel, self.velocities), (_ti_C, self.Cs)]:
            cur = field.to_numpy(); cur[sl] = arr.astype(np.float32); field.from_numpy(cur)

    kick_count = 0

    def kick(self):
        """Add a random impulse, capped so no particle exceeds the material's
        elastic wave speed c = √(μ/ρ).  Above c, stress can't propagate fast
        enough to hold the body together → surface particles escape.  This is
        the material-level CFL, not the grid CFL (which is much higher)."""
        self.kick_count += 1
        mag = np.random.uniform(4.0, 10.0)
        print(f"[kick #{self.kick_count}] mag={mag:.1f} m/s  material={self.mat['name']}")
        d   = np.random.randn(3); d /= np.linalg.norm(d)
        sl  = slice(self.start, self.end)
        cur_v = _ti_vel.to_numpy()
        cur_v[sl] += (d * mag).astype(np.float32)
        # Cap each particle's total speed at the material sound speed.
        # c = √(μ/ρ).  Free-fall (8.85 m/s) is well below c for all
        # materials; only accumulated kicks can breach it.
        mu_val  = float(self.mat["mu"])
        rho_val = float(self.mat["density"])
        c_sound = float(np.sqrt(mu_val / rho_val))
        speeds  = np.linalg.norm(cur_v[sl], axis=1)
        too_fast = speeds > c_sound
        if too_fast.any():
            scale = np.where(too_fast, c_sound / (speeds + 1e-8), 1.0)
            cur_v[sl] *= scale[:, None]
        _ti_vel.from_numpy(cur_v)
        self.velocities = cur_v[sl].astype(float)

    # ---- shape restore (sliced) ----------------------------------------

    def _shape_restore(self):
        """Procrustes shape matching restricted to [self.start, self.end)."""
        if self.shape_alpha == 0.0:
            return
        N = self.end - self.start
        _ti_shape_gather_com(self.start, self.end)
        com_sum = _ti_shape_com.to_numpy()
        _ti_shape_com.from_numpy((com_sum / N).astype(np.float32))
        _ti_shape_gather_H(self.start, self.end)
        H = _ti_shape_buf.to_numpy()
        U_s, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U_s.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U_s.T
        _ti_shape_R.from_numpy(R.astype(np.float32))
        if self.shape_alpha > 0.5:
            _ti_shape_apply_rigid(self.start, self.end, float(self.shape_alpha))
            _ti_zero_C(self.start, self.end)
        else:
            _ti_shape_apply(self.start, self.end, float(self.shape_alpha))

    # ---- per-frame advance ---------------------------------------------

    def _thermo_step(self):
        """One-way heat solve and latent-heat phase update for meltable UL bodies."""
        if (not THERMO_ENABLED or self.solver_name != "ul_mlsmpm"
                or "melt_temp" not in self.mat):
            return
        source_temp = float(HOT_FLOOR_TEMP) if HOT_FLOOR_TEMP is not None else (
            float(self.mat["melt_temp"]) + 60.0)
        solver_ul.heat_step(
            self.start, self.end, DT,
            conductivity=float(self.mat.get("conductivity", 0.0)),
            heat_capacity=float(self.mat.get("heat_capacity", 1.0)),
            density=float(self.mat.get("density", 1000.0)),
            floor_y=CONTACT_ZONE,
            source_temp=source_temp,
            n_iters=HEAT_DIFFUSE_ITERS,
            band_dx=HEAT_BAND_DX,
        )
        _ti_phase_update(
            self.start, self.end,
            float(self.mat["melt_temp"]),
            float(self.mat.get("latent_heat", 0.0)),
            float(self.mat.get("molten_kappa", self.mat.get("kappa", 0.0))),
            float(self.mat.get("heat_capacity", 1.0)),
            float(self.mat.get("molten_mu", 0.0)),
        )
        _ti_thermo_soften(
            self.start, self.end,
            float(self.mat["melt_temp"]),
            float(self.mat.get("softening_range", 20.0)),
        )
        if DEMO_SPREAD_ENABLED:
            _ti_molten_floor_spread(self.start, self.end, CONTACT_ZONE, DT, 12.0)

    def _finish_frame(self):
        """Post-substep work once per display frame (after SUBSTEPS substeps)."""
        is_fluid = self.mat["model"] == "fluid"
        self._thermo_step()
        if not is_fluid:
            self._shape_restore()
            if self.solver_name == "tl_apic":
                if self.fracture_active:
                    solver_tl.wls_recompute_fracture(self.start, self.end)
                else:
                    solver_tl.wls_recompute(self.start, self.end)
        if self.fracture_active and not is_fluid:
            _ti_damage_update(self.start, self.end,
                              self.yield_strain, self.damage_rate, DT)
            if self.solver_name == "tl_apic":
                # Damage is intentionally gated by tear mode; bond breaking
                # must follow the same gate.  Otherwise soft fracture-enabled
                # materials (jelly/bread) can lose WLS neighbours during an
                # ordinary floor drop even though damage stayed at zero.
                if _ti_tear_active[None] == 1:
                    solver_tl.bond_break_check(self.start, self.end,
                                               BOND_BREAK_STRETCH, BOND_BREAK_DMG_THRESH)
        _ti_polar_decomp()
        if not is_fluid and self.shape_alpha > 0.5:
            _ti_rigidify_F(self.start, self.end)
        if not is_fluid:
            _ti_plasticity_update(self.mu_scale, DT)
        _ti_update_covs()
        kappa = float(self.mat["kappa"])
        model = 0 if not is_fluid else 1
        _ti_diagnostics(kappa, self.mu_scale, model)
        _ti_continuum_state_update(self.start, self.end, kappa, self.mu_scale, model)

        sl = slice(self.start, self.end)
        self.positions        = _ti_pos.to_numpy()[sl].astype(float)
        self.velocities       = _ti_vel.to_numpy()[sl].astype(float)
        self.Fs               = _ti_F.to_numpy()[sl].astype(float)
        self.Us               = _ti_U.to_numpy()[sl].astype(float)
        self.covs             = _ti_covs.to_numpy()[sl].astype(float)
        self.diag_stress_norm = _ti_sn.to_numpy()[sl]
        self.diag_sigma_min   = _ti_sm.to_numpy()[sl]
        if THERMO_ENABLED and "melt_temp" in self.mat:
            temp = _ti_temp.to_numpy()[sl]
            phase = _ti_phase.to_numpy()[sl]
            heat_u = _ti_heat_U.to_numpy()[sl]
            self.thermo_molten   = int(np.count_nonzero(phase == 1))
            self.thermo_temp_min = float(temp.min()) if len(temp) else AMBIENT_TEMP
            self.thermo_temp_max = float(temp.max()) if len(temp) else AMBIENT_TEMP
            self.thermo_U_mean   = float(heat_u.mean()) if len(heat_u) else 0.0
        self.frame += 1

    def _sync_positions_only(self):
        """Lightweight GPU→CPU sync for mid-frame substep debugging."""
        sl = slice(self.start, self.end)
        self.positions  = _ti_pos.to_numpy()[sl].astype(float)
        self.velocities = _ti_vel.to_numpy()[sl].astype(float)

    def positions_torch(self):
        """Return this entity's positions as a torch CUDA tensor  -  NO CPU
        roundtrip.  Shares GPU memory with the Taichi field.  For feeding
        gs_render.render() directly without numpy bouncing through PCIe."""
        return _ti_pos.to_torch(device="cuda")[self.start:self.end]

    def covs_torch(self):
        """Return this entity's covariances as a torch CUDA tensor  -  no CPU
        roundtrip.  Pair with positions_torch() for zero-copy gs rendering."""
        return _ti_covs.to_torch(device="cuda")[self.start:self.end]

    def step(self):
        """Advance this entity by one display frame.  Substeps → shape restore →
        per-entity post-processing → sync slice back to CPU."""
        if not self.dropped:
            return
        self._substep_acc = 0
        dt_sub = DT / SUBSTEPS
        for _ in range(SUBSTEPS):
            self._substep(dt_sub)
        self._finish_frame()

    def substep_once(self):
        """Advance one MPM substep; run full frame post-process every SUBSTEPS substeps."""
        if not self.dropped:
            return
        self._substep(DT / SUBSTEPS)
        self._substep_acc += 1
        if self._substep_acc >= SUBSTEPS:
            self._substep_acc = 0
            self._finish_frame()
        else:
            self._sync_positions_only()

    def _substep(self, dt):
        """Solver-dispatched substep.  TL: rest-frame additive F.
        UL: world-frame multiplicative F.  Fluid: PBF (unchanged)."""
        if self.mat["model"] == "fluid":
            rho0        = float(self.pbf_rho0)
            restitution = float(self.mat.get("restitution", 0.05))
            damping     = float(self.mat["damping"])
            _ti_pbf_predict(dt)
            _ti_pbf_build_hash()
            for _ in range(_PBF_ITER):
                _ti_pbf_lambda(rho0)
                _ti_pbf_compute_dx(rho0)
                _ti_pbf_apply_dx()
            _ti_pbf_finalize(dt, restitution)
            _ti_pbf_viscosity(damping)
        else:
            kappa = float(self.mat["kappa"])
            # model int: 0=corotated, 1=fluid, 2=sand (Drucker-Prager)
            mat_model = self.mat["model"]
            model_int = 2 if mat_model == "sand" else (1 if mat_model == "fluid" else 0)
            if self.solver_name == "tl_apic":
                solver_tl.substep(
                    self.start, self.end, dt,
                    kappa=kappa, mu_scale=self.mu_scale, model=model_int,
                    floor_damp=self.floor_damp, friction=self.friction,
                    n_grid=self.mpm_n_grid, grid_h=self.grid_h,
                )
            else:  # ul_mlsmpm
                _do_project = (PRESSURE_PROJECT_ENABLED and THERMO_ENABLED and "melt_temp" in self.mat
                               and self.thermo_molten > 0)
                solver_ul.substep(
                    self.start, self.end, dt,
                    kappa=kappa, mu_scale=self.mu_scale, model=model_int,
                    floor_damp=self.floor_damp, friction=self.friction,
                    project=_do_project,
                    fluid_rho=float(self.mat.get("density", 1000.0)),
                )
            # Tear-mode anchors override velocity for top/bottom slabs.  Runs
            # after the MPM step so it has the final say on those particles'
            # trajectories.  No-op when _ti_tear_active is 0.
            if self.fracture_active:
                rest_slice = REST_POS[self.start:self.end]
                rest_y_lo  = float(rest_slice[:, 1].min())
                rest_y_hi  = float(rest_slice[:, 1].max())
                top_thr    = rest_y_hi - (rest_y_hi - rest_y_lo) * TEAR_SLAB_FRAC
                bot_thr    = rest_y_lo + (rest_y_hi - rest_y_lo) * TEAR_SLAB_FRAC
                _ti_tear_anchor(self.start, self.end, rest_y_lo, rest_y_hi,
                                top_thr, bot_thr, dt)

# ============================================================
# MESH BINDING
# ============================================================

def compute_bindings(mesh_verts, particle_pos, k=K_BIND):
    """
    Attach each mesh vertex to the k nearest particles using inverse-distance weights.

    Returns:
      idxs        (V, k)   -  which k particles each vertex is bound to
      weights     (V, k)   -  normalised IDW weights (sum to 1 per vertex)
      rest_offsets (V, 3)  -  the residual gap between the vertex and its weighted
                            particle average in the rest pose.  Stored so the vertex
                            can be reconstructed accurately even if it doesn't sit
                            exactly on top of any particle.
    """
    tree        = cKDTree(particle_pos)
    k           = min(k, len(particle_pos))   # scipy returns padded rows when k > n
    dists, idxs = tree.query(mesh_verts, k=k, workers=-1)
    w           = 1.0 / (dists + 1e-8)
    w          /= w.sum(axis=1, keepdims=True)   # normalise
    # weighted average of the bound particles' rest positions
    recon       = (w[:,:,None] * particle_pos[idxs]).sum(axis=1)
    # the offset is what's left over  -  preserved through deformation
    offsets     = mesh_verts - recon
    return idxs, w, offsets


def update_verts_local(idxs, weights, rest_offsets, particle_pos, particle_Us):
    """
    Recompute mesh vertex positions after the particles have moved.

    Each vertex position = (weighted average of bound particle positions)
                          + (locally stretched rest offset)

    The rest offset is transformed only by the symmetric stretch U (not the full F).
    Rotation is already captured by where the particles moved (recon), so applying
    the rotation part R of F = R*U again would double-count it and cause surface
    vertices to swivel in the wrong direction.
    """
    # weighted average of current particle positions
    recon   = (weights[:,:,None] * particle_pos[idxs]).sum(axis=1)
    # weighted average of local stretch matrices U (symmetric, rotation-free)
    U_local = (weights[:,:,None,None] * particle_Us[idxs]).sum(axis=1)
    # apply only the stretch to the rest offset: xoffset = U @ offset
    xoffset = np.einsum('vij,vj->vi', U_local, rest_offsets)
    return recon + xoffset


def update_normals(idxs, weights, covs):
    """
    Compute a surface normal for each mesh vertex from the Gaussian covariances.

    The smallest eigenvector of a covariance matrix points in the direction of
    least spread  -  which is the outward normal for a disc-shaped Gaussian lying
    flat on the ball surface.  We average the normals of the k bound particles
    to get a smooth interpolated normal at each vertex.
    """
    # smallest eigenvector of each covariance (eigh returns eigenvalues in ascending order)
    pn      = np.array([np.linalg.eigh(C)[1][:,0] for C in covs])
    # weighted average over the k bound particles
    normals = (weights[:,:,None] * pn[idxs]).sum(axis=1)
    # normalise to unit length
    norms   = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.where(norms > 1e-8, norms, 1.0)


# ============================================================
# GPU SKINNING HELPERS
# ============================================================

_skin_arrays: dict = {}   # holds ti.ndarray GPU buffers; rebuilt on shape-swap


def _rebuild_skin_arrays(idxs, weights, offsets):
    """Upload binding data to GPU-resident ti.ndarray objects.  Called once at startup
    and again on any shape-swap that changes the vertex count V."""
    V = len(idxs)
    K = idxs.shape[1]
    _skin_arrays["idxs"] = ti.ndarray(dtype=ti.i32, shape=(V, K))
    _skin_arrays["wts"]  = ti.ndarray(dtype=ti.f32, shape=(V, K))
    _skin_arrays["off"]  = ti.ndarray(dtype=ti.f32, shape=(V, 3))
    _skin_arrays["out"]  = ti.ndarray(dtype=ti.f32, shape=(V, 3))
    _skin_arrays["idxs"].from_numpy(np.ascontiguousarray(idxs,    dtype=np.int32))
    _skin_arrays["wts"].from_numpy( np.ascontiguousarray(weights, dtype=np.float32))
    _skin_arrays["off"].from_numpy( np.ascontiguousarray(offsets, dtype=np.float32))


def gpu_skin() -> np.ndarray:
    """Run GPU skinning kernel; returns (V, 3) float32 vertex positions."""
    sa = _skin_arrays
    _ti_skin_update(sa["out"].shape[0], sa["idxs"], sa["wts"], sa["off"], sa["out"])
    return sa["out"].to_numpy()


# ============================================================
# DRIFT MEASUREMENT
# ============================================================

def compute_iso_value(positions, covs, target_radius=None, n_samples=300):
    """
    Find the density value at the shape's surface in the REST configuration.

    For a sphere (target_radius given): samples on the known spherical surface.
    For arbitrary shapes (duck etc., target_radius=None): uses the outermost
    particles (beyond 75th percentile distance from centroid) as surface proxies.
    """
    if target_radius is not None:
        samples = fibonacci_sphere(n_samples, target_radius)
    else:
        centroid = positions.mean(axis=0)
        dists    = np.linalg.norm(positions - centroid, axis=1)
        samples  = positions[dists >= np.percentile(dists, 75)]

    density = np.zeros(len(samples))
    for p, C in zip(positions, covs):
        try:    C_inv = np.linalg.inv(C + 1e-6*np.eye(3))
        except: C_inv = np.eye(3)
        diff  = samples - p
        maha2 = (diff @ C_inv * diff).sum(axis=-1)
        density += np.exp(-0.5 * maha2)
    return float(density.mean())


def _density_field(positions, covs, grid_res=DRIFT_GRID_RES, padding=0.8):
    """
    Evaluate the Gaussian density at every point of a 3-D voxel grid.

    Returns the scalar field (grid_res³ array) and the world-space bounds
    of the grid (mins, maxs).  Used by measure_drift to run marching cubes.
    """
    mins = positions.min(axis=0) - padding
    maxs = positions.max(axis=0) + padding
    xs   = np.linspace(mins[0], maxs[0], grid_res)
    ys   = np.linspace(mins[1], maxs[1], grid_res)
    zs   = np.linspace(mins[2], maxs[2], grid_res)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
    grid_pts   = np.stack([gx, gy, gz], axis=-1)   # (R, R, R, 3)
    field      = np.zeros((grid_res,)*3, dtype=np.float32)
    for p, C in zip(positions, covs):
        try:    C_inv = np.linalg.inv(C + 1e-6*np.eye(3))
        except: C_inv = np.eye(3)
        diff  = grid_pts - p
        maha2 = (diff @ C_inv * diff).sum(axis=-1)
        field += np.exp(-0.5 * maha2)
    return field, mins, maxs




def measure_drift(skin_verts, cur_pos, cur_covs, iso_value):
    """
    Measure how far the visible mesh has drifted from the true Gaussian surface.

    Steps:
      1. Build a 3-D density field from current particle positions/covariances.
      2. Run marching cubes at iso_value to extract the "true" surface.
      3. For every visible mesh vertex, find its distance to the true surface.
      4. Return mean and max drift (in metres).

    A low drift means the mesh skin is faithfully following the physics.
    """
    field, mins, maxs = _density_field(cur_pos, cur_covs)
    if field.max() < iso_value:
        return None, None   # field too weak (e.g. ball has exploded off-screen)
    try:
        verts, _, _, _ = marching_cubes(field, level=iso_value)
    except (ValueError, RuntimeError):
        return None, None   # marching cubes failed (degenerate field)
    # convert voxel indices back to world-space metres
    spacing   = (maxs - mins) / (DRIFT_GRID_RES - 1)
    iso_world = verts * spacing + mins
    # nearest-neighbour distance from each mesh vertex to the true iso-surface
    dists, _  = cKDTree(iso_world).query(skin_verts, k=1)
    return float(dists.mean()), float(dists.max())

# ============================================================
# ELLIPSOID LINES
# ============================================================

def init_ellipsoid_pd(n):
    """
    Allocate a PyVista PolyData object to hold n*3 line segments (one per
    principal axis of each Gaussian ellipsoid).  Pre-building the connectivity
    array once avoids re-allocating it every frame.
    """
    lines = []
    for i in range(n * 3):
        lines.extend([2, i*2, i*2+1])   # each "line" is [num_pts, start_idx, end_idx]
    pd        = pv.PolyData()
    pd.points = np.zeros((n*6, 3), dtype=np.float32)
    pd.lines  = np.array(lines, dtype=np.int32)
    return pd


def fill_ellipsoid_points(positions, covs, scale=2.5):
    """
    Compute the endpoint pairs for all ellipsoid axis lines.

    For each particle i and each of its 3 principal axes j:
      axis vector = eigenvector_j * sqrt(eigenvalue_j) * scale
      start point = position - axis
      end   point = position + axis

    Stored as interleaved pairs: [p0_start, p0_end, p1_start, p1_end, …]
    matching the connectivity built in init_ellipsoid_pd.
    """
    n   = len(positions)
    pts = np.empty((n*6, 3), dtype=np.float32)
    for i, (pos, C) in enumerate(zip(positions, covs)):
        try:
            eigvals, eigvecs = np.linalg.eigh(C)   # eigh is faster than eig for symmetric matrices
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.ones(3), np.eye(3)
        eigvals = np.maximum(eigvals, 0)   # clamp negative eigenvalues (shouldn't occur but just in case)
        for j in range(3):
            ax             = eigvecs[:,j] * np.sqrt(eigvals[j]) * scale
            pts[i*6+j*2]   = pos - ax   # start of axis line
            pts[i*6+j*2+1] = pos + ax   # end of axis line
    return pts

# ============================================================
# VIEWER HELPERS  -  reference geometry, bookmarks, capture
# ============================================================

def _add_scene_reference(pl):
    """World axes (1 m) + ground scale bar for figure/video scale."""
    origin = (0.0, 0.0, 0.0)
    axis_specs = (
        ((1.0, 0.0, 0.0), "#cc3333", "X"),
        ((0.0, 1.0, 0.0), "#228833", "Y"),
        ((0.0, 0.0, 1.0), "#2255aa", "Z"),
    )
    for direction, color, _label in axis_specs:
        end = tuple(origin[i] + direction[i] for i in range(3))
        pl.add_mesh(pv.Line(origin, end), color=color, line_width=3)
    # 1 m scale bar on the floor grid (y slightly above ground plane)
    sb_a = (-5.5, 0.003, -5.5)
    sb_b = (-4.5, 0.003, -5.5)
    pl.add_mesh(pv.Line(sb_a, sb_b), color="#222222", line_width=4)
    pl.add_point_labels(
        [(-5.0, 0.003, -5.5)], ["1 m"],
        font_size=14, text_color="#222222", point_size=0, show_points=False,
        name="scale_bar_label",
    )


def _camera_snapshot(pl):
    cam = pl.camera
    return {
        "position":    tuple(float(x) for x in cam.position),
        "focal_point": tuple(float(x) for x in cam.focal_point),
        "view_up":     tuple(float(x) for x in cam.up),
        "view_angle":  float(cam.view_angle),
    }


def _camera_restore(pl, snap):
    pl.camera_position = [snap["position"], snap["focal_point"], snap["view_up"]]
    pl.camera.view_angle = snap["view_angle"]


def _sim_snapshot(sim):
    return {
        "positions":  sim.positions.copy(),
        "velocities": sim.velocities.copy(),
        "Fs":         sim.Fs.copy(),
        "Fp":         sim.Fp.copy(),
        "Cs":         sim.Cs.copy(),
        "covs":       sim.covs.copy(),
        "Us":         sim.Us.copy(),
        "dropped":    sim.dropped,
        "frame":      sim.frame,
        "mu_scale":   sim.mu_scale,
        "substep_acc": sim._substep_acc,
    }


def _sim_restore(sim, snap):
    sim.positions  = snap["positions"].copy()
    sim.velocities = snap["velocities"].copy()
    sim.Fs         = snap["Fs"].copy()
    sim.Fp         = snap["Fp"].copy()
    sim.Cs         = snap["Cs"].copy()
    sim.covs       = snap["covs"].copy()
    sim.Us         = snap["Us"].copy()
    sim.dropped    = snap["dropped"]
    sim.frame      = snap["frame"]
    sim.mu_scale   = snap["mu_scale"]
    sim._substep_acc = int(snap.get("substep_acc", 0))
    sim._upload_to_gpu()
    sl = slice(sim.start, sim.end)
    cur_cov = _ti_covs.to_numpy()
    cur_cov[sl] = sim.covs.astype(np.float32)
    _ti_covs.from_numpy(cur_cov)


# ============================================================
# MAIN
# ============================================================

def run():
    """
    Set up the 3-D window and run the main render loop.

    Sequence:
      1. Build the simulation and mesh binding.
      2. Create the PyVista window with ground plane, particle cloud,
         ellipsoid wireframes, and the skinned mesh.
      3. Register keyboard callbacks for user interaction.
      4. Loop: process events → step physics → update visuals → render.
    """
    # Create sim entities.  In collide mode: two UL entities with separate
    # particle slices on the same world grid (collision emerges from shared P2G).
    # In single mode: one entity spanning all particles.
    mat_key = _CLI.material or "1"
    if SCENE == "collide":
        sims = []
        for ent_id, (slv, spawn) in enumerate(zip(ENTITY_SOLVERS, SPAWN_POSITIONS)):
            s = GaussianBallSim(
                material_key=mat_key, entity_id=ent_id,
                start=ent_id * N_PER_ENTITY, end=(ent_id + 1) * N_PER_ENTITY,
                solver_name=slv, spawn_position=spawn,
            )
            sims.append(s)
        sim = sims[0]  # primary entity for UI / visualization reference
    else:
        sim = GaussianBallSim(material_key=mat_key)
        sims = [sim]

    # Multi-material: overwrite top-half particles with a second material's μ/λ.
    # The particle struct is unchanged  -  only the per-particle material fields
    # differ, demonstrating "material identity per particle" from the paper.
    _multi_mat_key = getattr(_CLI, "multi_material", None)
    if _multi_mat_key and _multi_mat_key in MATERIALS:
        mat_top = MATERIALS[_multi_mat_key]
        rest_y  = _ti_rest.to_numpy()[sim.start:sim.end, 1]
        y_mid   = float(rest_y.mean())
        top_mask = rest_y > y_mid
        N = sim.end - sim.start
        sl = slice(sim.start, sim.end)
        # Overwrite ALL per-particle material fields for the top half  - 
        # not just μ/λ but yield, plasticity mode, damping, density, etc.
        # This gives true multi-material: different physics behavior per
        # particle region, not just different stiffness numbers.
        def _overwrite(field, value):
            cur = field.to_numpy(); cur[sl][top_mask] = value; field.from_numpy(cur)
        _overwrite(_ti_mu,      float(mat_top["mu"]))
        _overwrite(_ti_lam,     float(mat_top["lam"]))
        _overwrite(_ti_yield,   float(mat_top.get("yield_stress", 1e9)))
        _overwrite(_ti_relax,   float(mat_top.get("relax_time", 0.0)))
        _overwrite(_ti_damping, float(mat_top.get("damping", 0.0)) /
                                max(float(mat_top.get("density", 1000.0)) * V0 if not isinstance(V0, np.ndarray) else 1.0, 1e-8))
        j2_val = mat_top.get("plastic_j2", 0)
        _overwrite(_ti_plastic_j2, int(j2_val) if isinstance(j2_val, (int, float)) else (1 if j2_val else 0))
        _overwrite(_ti_tension_cap, float(mat_top.get("tension_stretch_cap", 0.0)))
        _overwrite(_ti_sr_gamma,    float(mat_top.get("strain_rate_gamma", 0.0)))
        # Drucker-Prager alpha from friction angle
        phi_deg = float(mat_top.get("friction_angle", 0.0))
        if phi_deg > 0.0:
            dp_a = np.sin(np.radians(phi_deg)) / (np.sqrt(3.0) * np.cos(np.radians(phi_deg)))
        else:
            dp_a = 0.0
        _overwrite(_ti_dp_alpha, dp_a)
        # Per-particle constitutive law  -  this is the key: top half can run
        # a DIFFERENT stress equation (e.g. Drucker-Prager) than bottom half
        # (e.g. corotated).  True multi-law, not just multi-parameter.
        mat_model_str = mat_top.get("model", "corotated")
        model_int_top = 2 if mat_model_str == "sand" else (1 if mat_model_str == "fluid" else 0)
        _overwrite(_ti_model, model_int_top)
        # Mass from density
        density_top = float(mat_top.get("density", 1000.0))
        if isinstance(V0, np.ndarray):
            mass_top = (density_top * V0[top_mask]).astype(np.float32)
            cur_m = _ti_m.to_numpy(); cur_m[sl][top_mask] = mass_top; _ti_m.from_numpy(cur_m)
        else:
            _overwrite(_ti_m, density_top * V0)
        n_top = int(top_mask.sum())
        print(f"[multi-material] top half ({n_top}/{N} particles): "
              f"{mat_top['name']} mu={mat_top['mu']}, lam={mat_top['lam']}, "
              f"yield={mat_top.get('yield_stress', 1e9)}, "
              f"j2={mat_top.get('plastic_j2', 0)}")
        print(f"[multi-material] bottom half ({N-n_top}/{N} particles): "
              f"{sim.mat['name']} mu={sim.mat['mu']}, lam={sim.mat['lam']}")

    # 3DGS rasterized view (Piece 2).  Only valid for PLY scenes  -  the
    # opacity + SH tensors come from the trained 3DGS model.
    gs_renderer = None
    if GS_RENDER_ENABLED:
        import cv2 as _cv2_gs
        from gs_render import GSRenderer
        if SHAPE == "ply":
            _opac = _PLY_OPACITY_PER_ENTITY[0]
            _shs  = _PLY_SHS_PER_ENTITY[0]
        else:
            # Synthesize appearance for procedural shapes.
            #
            # A real 3DGS scene has Gaussians as a thin layer of FLAT DISCS
            # on surfaces, not isotropic blobs filling a volume.  Two fixes:
            #
            # 1. Override _ti_rcov with surface-aligned flat discs (σ_r ≪ σ_t)
            #    via make_base_covs.  Physics never reads _ti_rcov  -  it's
            #    purely for the Σ = F·Σ₀·F^T rendering update.
            #
            # 2. Only the outer shell (top 15% by radius) gets opacity; the
            #    interior is invisible.  This prevents alpha-saturation from
            #    hundreds of overlapping interior Gaussians.
            _n_gs = N_PARTICLES
            _sh_c0 = 0.28209479
            _sh_c1 = 0.48860251

            # Parse hex color → [0,1] RGB
            _hex = sim.mat.get("mesh_color", "#888888").lstrip("#")
            _rgb = np.array([int(_hex[i:i+2], 16) / 255.0 for i in (0, 2, 4)])

            # Override rest covariance with surface-aligned flat discs.
            # σ_t = 0.18 (tangent, wide), σ_r = 0.02 (radial, thin).
            # These are MUCH larger than the grid-fill isotropic σ≈0.046
            # and anisotropic  -  gives a smooth visual shell.
            _rest_slice = REST_POS[:_n_gs]
            _render_covs = make_base_covs(_rest_slice,
                                          radial_spread=0.02,
                                          tangent_spread=0.18)
            cur_rcov = _ti_rcov.to_numpy()
            cur_rcov[:_n_gs] = _render_covs.astype(np.float32)
            _ti_rcov.from_numpy(cur_rcov)
            _ti_update_covs()   # recompute Σ = F·Σ₀·F^T with new rcov

            # Opacity: only the outer 15% shell is visible.
            _center = _rest_slice.mean(axis=0)
            _dist = np.linalg.norm(_rest_slice - _center, axis=1)
            _max_r = float(_dist.max()) + 1e-6
            _depth_frac = _dist / _max_r
            _opac_raw = np.clip((_depth_frac - 0.85) / 0.15, 0.0, 1.0)
            _opac = (_opac_raw * 0.95)[:, None].astype(np.float32)

            # SH: DC = base color, band-1 = directional light (top-down)
            _shs = np.zeros((_n_gs, 16, 3), dtype=np.float32)
            _shs[:, 0, :] = ((_rgb - 0.5) / _sh_c0).astype(np.float32)
            _shs[:, 2, :] = (0.5 / _sh_c1)    # Y₁⁰ (y-axis top light)
            _shs[:, 1, :] = (0.1 / _sh_c1)    # slight x fill
            _shs[:, 3, :] = (0.05 / _sh_c1)   # slight z fill

            _n_surface = int((_opac_raw > 0.01).sum())
            print(f"[gs-render] Surface shell: {_n_surface}/{_n_gs} visible "
                  f"particles, flat-disc covs (σ_r=0.02 σ_t=0.18), "
                  f"color={_rgb.round(2)}")
        gs_renderer = GSRenderer(_opac, _shs, bg=(1.0, 1.0, 1.0))
        if VIDEO_OUT:
            # Write to MP4  -  no display, no per-frame imshow latency.
            _fourcc = _cv2_gs.VideoWriter_fourcc(*"mp4v")
            video_writer = _cv2_gs.VideoWriter(VIDEO_OUT, _fourcc, 30.0, (800, 800))
            print(f"[gs-render] Recording to {VIDEO_OUT} @ 30fps")
        else:
            video_writer = None
            _cv2_gs.namedWindow("3DGS substrate", _cv2_gs.WINDOW_NORMAL)
            _cv2_gs.resizeWindow("3DGS substrate", 800, 800)
        print(f"[gs-render] Rasterizer initialised on {gs_renderer.N} gaussians")

    # Compute the Gaussian density at the shape surface in rest pose.
    if SHAPE == "sphere":
        iso_value = compute_iso_value(REST_POS, sim.rest_covs, target_radius=BALL_RADIUS)
        sphere    = pv.Sphere(radius=BALL_RADIUS, theta_resolution=22, phi_resolution=22)
        mesh_verts_rest = np.array(sphere.points, dtype=float)
        duck_mesh = cuboid_mesh = bunny_mesh = custom_mesh = None
    elif SHAPE == "cuboid":
        iso_value = compute_iso_value(REST_POS, sim.rest_covs, target_radius=None)
        _hx, _hy, _hz = CUBOID_HALF_EXTENTS
        cuboid_mesh = pv.Box(bounds=(-float(_hx), float(_hx), -float(_hy), float(_hy),
                                      -float(_hz), float(_hz)))
        mesh_verts_rest = np.array(cuboid_mesh.points, dtype=float)
        duck_mesh = bunny_mesh = custom_mesh = None
    elif SHAPE == "bunny":
        iso_value = compute_iso_value(REST_POS, sim.rest_covs, target_radius=None)
        bunny_mesh     = _BUNNY_MESH.copy()
        mesh_verts_rest = np.array(bunny_mesh.points, dtype=float)
        duck_mesh = cuboid_mesh = custom_mesh = None
    elif SHAPE == "custom":
        iso_value = compute_iso_value(REST_POS, sim.rest_covs, target_radius=None)
        custom_mesh     = _CUSTOM_MESH.copy()
        mesh_verts_rest = np.array(custom_mesh.points, dtype=float)
        duck_mesh = bunny_mesh = cuboid_mesh = None
    elif SHAPE == "ply":
        iso_value = 0.0
        mesh_verts_rest = REST_POS.astype(float).copy()
        duck_mesh = pv.PolyData(REST_POS.astype(np.float32))
        cuboid_mesh = bunny_mesh = custom_mesh = None
    else:
        iso_value = compute_iso_value(REST_POS, sim.rest_covs, target_radius=None)
        # Extract marching-cubes mesh from the rest-pose Gaussian density field
        field, mins, maxs = _density_field(REST_POS, sim.rest_covs, grid_res=48, padding=0.5)
        spacing = (maxs - mins) / (np.array(field.shape) - 1)
        from skimage.measure import marching_cubes as _mc
        verts_mc, faces_mc, _, _ = _mc(field, level=iso_value, spacing=tuple(spacing))
        verts_mc += mins
        faces_flat = np.hstack([np.full((len(faces_mc),1), 3), faces_mc]).ravel()
        duck_mesh  = pv.PolyData(verts_mc.astype(np.float32), faces_flat)
        mesh_verts_rest = verts_mc.astype(float)
        cuboid_mesh = bunny_mesh = custom_mesh = None

    # Bind each vertex to its nearest particles (IDW weights + rest offsets).
    # Skipped entirely under --no-skin: no IDW binding, no GPU skin arrays, no
    # mesh→particle relationship at all.  The substrate runs particle-only.
    if not NO_SKIN:
        _bind = list(compute_bindings(mesh_verts_rest, REST_POS))  # [idxs, weights, rest_offsets]
        _rebuild_skin_arrays(_bind[0], _bind[1], _bind[2])
        # Snapshot original idxs+weights for fracture-aware skin rebinding  - 
        # never overwritten on shape swap so we always know the "intact" baseline.
        _frac_orig_idxs    = _bind[0].copy()
        _frac_orig_weights = _bind[1].copy()
        _frac_last_intact_count = int(N_PARTICLES * K_PHYS)   # all bonds intact at start
        err = np.linalg.norm(
            mesh_verts_rest - (_bind[1][:,:,None]*REST_POS[_bind[0]]).sum(axis=1), axis=1)
    else:
        _bind = [None, None, None]
        err = np.zeros(1)   # no skin -> no binding error to report
    print("=" * 60)
    print(f"Inverted MaGS -- MPM Constitutive Laws  [{SHAPE.upper()}]")
    print(f"  Mesh vertices      : {len(mesh_verts_rest)}")
    print(f"  Gaussian particles : {N_PARTICLES}  (K_bind={K_BIND})")
    print(f"  MPM grid nodes     : {sim.mpm_n_grid}  (h={MPM_GRID_H:.3f})")
    print(f"  ISO value          : {iso_value:.4f}")
    if isinstance(V0, np.ndarray):
        print(f"  V0 (per-particle)  : mean={float(V0.mean()):.6f}  "
              f"total={float(V0.sum()):.4f}  (grid-fill)")
    else:
        print(f"  V0                 : {V0:.6f}")
    print(f"  Mean IDW error     : {err.mean():.5f}")
    print(f"  Max  IDW error     : {err.max():.5f}")
    print("=" * 60)

    # --- pyvista scene setup ---
    pv.global_theme.allow_empty_mesh = True
    pl = pv.Plotter(window_size=(1280, 720),
                    title="GaussianFlesh  -  Universal Physics Substrate")
    pl.set_background("#f0f2f5")  # clean off-white lab

    # Ground  -  white with subtle grid
    ground = pv.Plane(center=(0, 0, 0), direction=(0,1,0), i_size=14, j_size=14)
    pl.add_mesh(ground, color="#e8eaed", opacity=1.0, show_edges=False)

    # Grid lines on floor
    for i in np.arange(-6, 7, 1.0):
        l1 = pv.Line((i, 0.001, -6), (i,  0.001, 6))
        l2 = pv.Line((-6, 0.001, i), (6, 0.001, i))
        pl.add_mesh(l1, color="#d0d4da", line_width=1)
        pl.add_mesh(l2, color="#d0d4da", line_width=1)

    # Back wall
    back_wall = pv.Plane(center=(0, 4, -6), direction=(0,0,1), i_size=14, j_size=10)
    pl.add_mesh(back_wall, color="#edf0f4", opacity=1.0)

    # Lighting  -  bright lab feel
    pl.remove_all_lights()
    pl.add_light(pv.Light(position=(4, 10, 6),  focal_point=(0,0,0), intensity=0.8))
    pl.add_light(pv.Light(position=(-4, 8, -2), focal_point=(0,0,0), intensity=0.5))
    pl.add_light(pv.Light(position=(0, 2, 8),   focal_point=(0,0,0), intensity=0.3))

    _add_scene_reference(pl)

    # Gripper plate  -  flat box visualising the ceiling contact plane
    _gripper_y_vis = [9999.0]   # current bottom-face y position
    _gripper_active = [False]
    # Box centred at origin, 0.15 thick  -  translated via actor transform each frame
    _gripper_mesh = pv.Box(bounds=(-1.2, 1.2, -0.075, 0.075, -1.2, 1.2))
    gripper_actor = pl.add_mesh(_gripper_mesh, color="#cc4444", opacity=0.7)
    gripper_actor.SetVisibility(False)

    # ── Webcam hand tracker (Tier 3 capsule-skeleton collider) ───────────────
    hand_tracker      = None
    hand_actors       = []     # list of (lines_actor, joints_actor) per hand
    hand_line_meshes  = []     # list of pv.PolyData per hand (mutable points)
    hand_joint_clouds = []     # list of pv.PolyData per hand (joint dots)
    # Previous-frame world-space hand positions, used for finite-difference velocity.
    # Parked at y=9999 by default so velocity is zero before any detection.
    _prev_hand_world  = np.zeros((N_HAND_PTS, 3), dtype=np.float32)
    _prev_hand_world[:, 1] = 9999.0
    # Frames since each hand was last detected.  Used for hysteresis: keep hand
    # visible for HAND_HOLD_FRAMES after a loss before parking it.
    _hand_lost_age     = [HAND_HOLD_FRAMES + 1] * N_HANDS_MAX   # both start "lost"
    _hand_was_detected = [False] * N_HANDS_MAX
    if HAND_ENABLED:
        import cam_input
        hand_tracker = cam_input.HandTracker(max_hands=N_HANDS_MAX)
        hand_tracker.start()
        print(f"  Webcam hand tracker: starting (downloading model if needed)")
        # Build one line-mesh + joint-cloud per hand.  Points are parked far above
        # the scene until detections arrive; mesh.points is mutated each frame.
        hand_colors = ["#00d4ff", "#ff66cc"]   # cyan = hand 0, magenta = hand 1
        for h in range(N_HANDS_MAX):
            base = h * N_HAND_PTS_PER
            init_pts = np.zeros((N_HAND_PTS_PER, 3), dtype=np.float32)
            init_pts[:, 1] = 9999.0
            # Bone connectivity as PyVista line cells: [2, a, b, 2, a, b, ...]
            line_cells = np.empty(N_HAND_BONES_PER * 3, dtype=np.int64)
            for b_idx, (a, bb) in enumerate(HAND_CONNECTIONS):
                line_cells[b_idx*3 + 0] = 2
                line_cells[b_idx*3 + 1] = a
                line_cells[b_idx*3 + 2] = bb
            lm = pv.PolyData(init_pts)
            lm.lines = line_cells
            hand_line_meshes.append(lm)
            la = pl.add_mesh(lm, color=hand_colors[h], line_width=4, opacity=1.0)
            # Joint dots  -  same point cloud rendered with larger spheres
            jc = pv.PolyData(init_pts.copy())
            hand_joint_clouds.append(jc)
            ja = pl.add_mesh(jc, color=hand_colors[h], point_size=10,
                              render_points_as_spheres=True, show_scalar_bar=False)
            hand_actors.append((la, ja))

    # ── Xbox controller init ──────────────────────────────────────────────────
    pygame.init()
    pygame.joystick.init()
    joy = pygame.joystick.Joystick(0) if pygame.joystick.get_count() > 0 else None
    if joy:
        joy.init()
        print(f"  Controller: {joy.get_name()}")
    else:
        print("  No controller found  -  keyboard only")

    # Hammer visual  -  wide flat head sits ABOVE the contact point, handle hangs below.
    # Contact sphere is at the bottom face of the head  -  swinging down drives the
    # head into the target from above, which is the natural "swing down" motion.
    # Head: wide in X/Z (striking face), flat in Y.
    _hammer_head   = pv.Box(bounds=(-0.17, 0.17, 0.0, 0.16, -0.14, 0.14))
    _hammer_handle = pv.Cylinder(center=(0, 0.0, 0), direction=(0, 1, 0),
                                  radius=0.032, height=0.72, resolution=12)
    hammer_actor_h  = pl.add_mesh(_hammer_head,   color="#1a1a1a", opacity=1.0)
    hammer_actor_hd = pl.add_mesh(_hammer_handle, color="#6b3d1e", opacity=1.0)

    # Hammer state  -  position and float height
    _hx   = [0.0]
    _hy   = [6.5]   # start high so there is room to swing down
    _hz   = [0.0]
    _hvx  = [0.0]
    _hvy  = [0.0]
    _hvz  = [0.0]

    def _move_hammer(x, y, z, vx=0.0, vy=0.0, vz=0.0):
        _hx[0], _hy[0], _hz[0] = x, y, z
        _ti_hammer_pos[None] = ti.Vector([x, y, z])
        _ti_hammer_vel[None] = ti.Vector([vx, vy, vz])
        # head body above contact sphere, handle hangs below
        hammer_actor_h.SetPosition(x, y, z)
        hammer_actor_hd.SetPosition(x, y - 0.36, z)

    _move_hammer(0.0, 6.5, 0.0)

    # Particle dots coloured by σ_min (smallest singular value of F).
    # σ_min = 1.0 at rest (no deformation) → green.
    # σ_min approaching 0.2 (SVD clamp threshold) → yellow/orange.
    # σ_min ≤ 0.2 (clamp is active, near-inversion) → red.
    # clim [0, 1]: anything ≥ 1 is fully safe (green); 0 is worst case (red).
    _init_pos = np.concatenate([s.positions for s in sims], axis=0).astype(np.float32)
    part_cloud = pv.PolyData(_init_pos)
    part_cloud.point_data["sigma_min"] = np.ones(len(_init_pos), dtype=np.float32)
    # When --no-skin is set, the mesh skin is hidden and particles are the
    # rendering primitive; otherwise particles are invisible underlay for diagnostics.
    # Show particles directly if --no-skin OR if fracture is active (the
    # damage colour signal lives on the particle cloud).
    _particle_opacity = 1.0 if (NO_SKIN or sim.fracture_active) else 0.0
    _skin_opacity     = 0.0 if NO_SKIN else 1.0
    # Thermo runs: colour particles by temperature (inferno: dark=cold, bright=hot)
    # so the melt front is visible.  Otherwise colour by σ_min deformation diagnostic.
    _thermo_view = THERMO_ENABLED and "melt_temp" in sim.mat
    if _thermo_view:
        _src_t = float(HOT_FLOOR_TEMP) if HOT_FLOOR_TEMP is not None else (
            float(sim.mat["melt_temp"]) + 60.0)
        part_cloud.point_data["temp"] = np.full(len(_init_pos), AMBIENT_TEMP, dtype=np.float32)
        part_actor = pl.add_mesh(part_cloud, scalars="temp", cmap="inferno",
                                 clim=[AMBIENT_TEMP, _src_t], point_size=12,
                                 render_points_as_spheres=True, opacity=_particle_opacity,
                                 show_scalar_bar=True, scalar_bar_args={"title": "Temp (deg)"})
    else:
        part_actor = pl.add_mesh(part_cloud, scalars="sigma_min", cmap="RdYlGn",
                                 clim=[0.0, 1.0], point_size=12,
                                 render_points_as_spheres=True, opacity=_particle_opacity,
                                 show_scalar_bar=False)

    # the actual visible ball mesh (skinned to particles)
    if SHAPE == "duck":
        _skin_src = duck_mesh
    elif SHAPE == "cuboid":
        _skin_src = cuboid_mesh
    elif SHAPE == "bunny":
        _skin_src = bunny_mesh
    elif SHAPE == "custom":
        _skin_src = custom_mesh
    elif SHAPE == "ply":
        _skin_src = duck_mesh
    else:
        _skin_src = sphere
    skin = _skin_src.copy()
    # With --no-skin: skip the GPU skin compute AND the actor entirely. The
    # particle cloud is the sole rendering primitive; the simulator is fully
    # independent of the mesh per-frame.  skin object stays defined (as a
    # dummy) so other code paths that reference it don't crash, but it's
    # never updated and never added to the scene.
    if not NO_SKIN:
        skin.points = gpu_skin()
        skin_actors = [pl.add_mesh(skin, color=MATERIALS["1"]["mesh_color"],
                                    style="surface", show_edges=False, opacity=_skin_opacity)]
    else:
        skin_actors = [None]

    # Fluid surface actor  -  always present in the scene, visibility toggled.
    # Mesh data is replaced in-place via the VTK mapper so there is never a
    # blank frame between removing the old surface and adding the new one.
    _fluid_pd   = pv.PolyData()   # persistent empty mesh  -  filled each frame
    fluid_actor = [pl.add_mesh(_fluid_pd, color=MATERIALS["4"]["mesh_color"],
                                opacity=0.80, smooth_shading=True, show_edges=False)]
    fluid_actor[0].SetVisibility(False)   # hidden until fluid is active + dropped

    def _update_fluid_surface():
        """PBF fluid: particles ARE the rendering primitive  -  no mesh extraction.
        The particle cloud (part_cloud) is always visible; we just ensure the
        marching-cubes actor stays hidden."""
        fluid_actor[0].SetVisibility(False)

    # --- HUD text overlay ---
    txt      = [None]
    paused   = [False]
    hud_on   = [True]
    drift_st = {"mean": None, "max": None, "peak_mean": 0.0, "peak_max": 0.0}
    playback_speed = [1.0]          # wall-clock: 0.25 = 4× slower viewing
    substep_mode   = [False]        # one MPM substep per display frame
    step_once      = [False]        # N while paused → one substep
    bookmark       = [None]         # saved camera + sim state
    recording      = [False]
    record_file    = [None]
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    # Persistent high-contrast frame counter in the upper-right, independent of
    # the H toggle so it stays visible even when the HUD is hidden.  Replaced
    # each frame via the `name` parameter (pyvista swaps the actor in-place).
    pl.add_text("frame 0", position=(500, 660), color="blue",
                font_size=28, name="frame_counter")

    def redraw_text():
        """Rebuild the on-screen stats text (material name, drift, stretch, controls)."""
        if txt[0] is not None:
            pl.remove_actor(txt[0])
            txt[0] = None
        if not hud_on[0]:
            return
        is_fl = sim.mat["model"] == "fluid"
        dm   = drift_st["mean"]
        dstr = (f"PBF fluid  -  particle splats  "
                f"ρ₀={sim.pbf_rho0:.4f}  iter={_PBF_ITER}  "
                f"ε={_PBF_EPS:.0f}  xsph={_PBF_XSPH:.3f}") \
               if is_fl else \
               ((f"Drift  now={dm:.4f}  max={drift_st['max']:.4f}"
                 f"  PEAK={drift_st['peak_mean']:.4f}")
                if dm is not None else "Drift: measuring...")
        if is_fl:
            rho_r = sim.diag_rho_mean / max(sim.pbf_rho0, 1e-6)
            ustr  = (f"PBF density  ρ_mean={sim.diag_rho_mean:.4f}  "
                     f"ρ₀={sim.pbf_rho0:.4f}  ratio={rho_r:.3f}  "
                     f"(1.0=rest, >1=compressed, <1=droplets)")
        else:
            uyy  = sim.Us[:, 1, 1]
            ustr = (f"U_yy (pure stretch)  min={uyy.min():.3f}  "
                    f"max={uyy.max():.3f}  spread={uyy.max()-uyy.min():.3f}")
        sm   = sim.diag_sigma_min
        sstr = (f"σ_min  min={sm.min():.3f}  mean={sm.mean():.3f}  "
                f"[red dots = clamped: {(sm < 0.2).sum()} particles]")
        mat_color = sim.mat.get("mesh_color", "#ffffff")
        if paused[0]:
            status = "PAUSED"
        elif substep_mode[0]:
            status = f"SUBSTEP ({sim._substep_acc}/{SUBSTEPS})"
        elif sim.dropped:
            status = "SIMULATING"
        else:
            status = "READY"
        rec_str = f"REC → {record_file[0]}" if recording[0] else "REC off"
        txt[0] = pl.add_text(
            f"  GAUSSIANFLESH  //  UNIVERSAL PHYSICS SUBSTRATE\n"
            f"  {'─' * 52}\n"
            f"  MATERIAL     {sim.material_name.upper():<12}  "
            f"MODEL  {sim.mat['model'].upper():<12}  "
            f"STATUS  {status}\n"
            f"  PLAYBACK  {playback_speed[0]:.2f}×   {rec_str}\n"
            f"  {'─' * 52}\n"
            f"  {ustr}\n"
            f"  {dstr}\n"
            f"  {sstr}\n"
            f"  {'─' * 52}\n"
            f"  [1]Rubber  [2]Metal  [3]Jelly  [4]Fluid  [5]Wood  [6]Gas  [C]Clay  [8]Flesh\n"
            f"  [SPC]Drop  [R]Reset  [P]Pause  [N]Substep  [[/]]Speed  [0/9]Bookmark\n"
            f"  [F]Screenshot  [V]Record  [K]Kick  [S]Shape  [H]HUD  [7]Cam",
            position="upper_left", font_size=9, color="#1a2540",
            font="courier",
        )

    pl.camera_position = [(0, 2, 8), (0, 1, 0), (0, 1, 0)]
    redraw_text()
    pl.show(interactive_update=True)

    # --- keyboard callback ---
    def _on_key(obj, event):
        """Handle a key press from the VTK interactor."""
        global SHAPE, V0, MPM_GRID_H, _BUNNY_MESH, _CUSTOM_MESH
        k = obj.GetKeySym()
        if   k in ("q","Q","Escape"): running[0] = False
        elif k == "space":            [s.drop() for s in sims]
        elif k in ("r","R"):
            [s.reset() for s in sims]
            drift_st["peak_mean"] = 0.0
            drift_st["peak_max"]  = 0.0
            part_cloud.points = sim.positions.astype(np.float32)
            if sim.mat["model"] == "fluid":
                fluid_actor[0].SetVisibility(False)   # hide until re-dropped
            else:
                iso_value_ref[0] = compute_iso_value(REST_POS, sim.rest_covs,
                                                      target_radius=BALL_RADIUS if SHAPE=="sphere" else None)
                if not NO_SKIN:
                    skin.points = gpu_skin()
        elif k in ("p","P"):  paused[0] = not paused[0]
        elif k in ("h","H"):  hud_on[0] = not hud_on[0]
        elif k in ("k","K"):  sim.kick()
        elif k in ("g", "G", "bracketright", "]"):
            # Activate press  -  starts 1.5m above current ball top and descends
            if not _gripper_active[0]:
                _gripper_active[0] = True
                all_y = np.concatenate([s.positions[:, 1] for s in sims])
                ball_top = float(all_y.max()) + 1.5
                _gripper_y_vis[0] = ball_top
                gripper_actor.SetPosition(0, ball_top + 0.075, 0)
                gripper_actor.SetVisibility(True)
        elif k in ("bracketleft", "["):
            # Deactivate press  -  pull plate away
            _gripper_active[0] = False
            _gripper_y_vis[0] = 9999.0
            _ti_gripper_y[None] = 9999.0
            gripper_actor.SetVisibility(False)
        elif k in ("s","S"):
            # --- cycle shape: sphere → duck → cuboid [→ custom] → sphere ---
            _shape_order = ["sphere", "duck", "cuboid", "bunny"]
            if CUSTOM_OBJ_PATH:
                _shape_order.append("custom")
            _cur = SHAPE if SHAPE in _shape_order else "sphere"
            SHAPE = _shape_order[(_shape_order.index(_cur) + 1) % len(_shape_order)]
            if SHAPE == "bunny":
                if _BUNNY_MESH is None:
                    _BUNNY_MESH = load_centered_mesh(BUNNY_OBJ_PATH, BUNNY_OBJ_SCALE, "bunny")
                REST_POS[:], V0, MPM_GRID_H, _spawn_y = _mesh_shape_rest(_BUNNY_MESH, N_PARTICLES)
                BALL_START[:] = [0.0, _spawn_y, 0.0]
                new_mesh      = _BUNNY_MESH.copy()
            elif SHAPE == "custom":
                if _CUSTOM_MESH is None:
                    _CUSTOM_MESH = load_centered_mesh(CUSTOM_OBJ_PATH, CUSTOM_OBJ_SCALE, "custom")
                REST_POS[:]   = sample_mesh_volume(_CUSTOM_MESH, N_PARTICLES)
                _vol          = float(_CUSTOM_MESH.volume) if _CUSTOM_MESH.volume > 0 else float(
                                    np.prod(np.array(_CUSTOM_MESH.bounds)[1::2] - np.array(_CUSTOM_MESH.bounds)[0::2]))
                V0            = _vol / N_PARTICLES
                _ext          = np.array(_CUSTOM_MESH.bounds)[1::2] - np.array(_CUSTOM_MESH.bounds)[0::2]
                MPM_GRID_H    = float(max(_ext)) / 12.0
                BALL_START[:] = [0.0, float(_CUSTOM_MESH.bounds[3]) + 4.0, 0.0]
                new_mesh      = _CUSTOM_MESH.copy()
            elif SHAPE == "duck":
                REST_POS[:]   = sample_duck_volume(N_PARTICLES)
                _sv = (1.0*0.65*0.85 + 0.45**3 + 0.22*0.12*0.28 + 0.28*0.30*0.22) * (4.0/3.0)*np.pi
                V0            = _sv / N_PARTICLES
                MPM_GRID_H    = 0.35
                BALL_START[:] = [0.0, 5.5, 0.0]
                # rebuild marching-cubes mesh for duck
                s = mat["cov_spread"] if (mat := sim.mat) else 0.12
                _new_covs  = make_base_covs(REST_POS, s, s)
                _field, _mins, _maxs = _density_field(REST_POS, _new_covs, grid_res=48, padding=0.5)
                _iso  = compute_iso_value(REST_POS, _new_covs, target_radius=None)
                _sp   = (_maxs - _mins) / (np.array(_field.shape) - 1)
                from skimage.measure import marching_cubes as _mc
                _v, _f, _, _ = _mc(_field, level=_iso, spacing=tuple(_sp))
                _v += _mins
                _ff = np.hstack([np.full((len(_f),1),3), _f]).ravel()
                new_mesh = pv.PolyData(_v.astype(np.float32), _ff)
            elif SHAPE == "cuboid":
                REST_POS[:]   = sample_cuboid_volume(N_PARTICLES, CUBOID_HALF_EXTENTS)
                _hx, _hy, _hz = CUBOID_HALF_EXTENTS
                V0            = float((2.0 * _hx) * (2.0 * _hy) * (2.0 * _hz) / N_PARTICLES)
                MPM_GRID_H    = float(min(CUBOID_HALF_EXTENTS) / 4.0)
                BALL_START[:] = [0.0, 4.0, 0.0]
                new_mesh = pv.Box(bounds=(-float(_hx), float(_hx), -float(_hy), float(_hy),
                                      -float(_hz), float(_hz)))
            else:
                REST_POS[:]   = sample_sphere_volume(N_PARTICLES, BALL_RADIUS)
                V0            = (4.0/3.0 * np.pi * BALL_RADIUS**3) / N_PARTICLES
                MPM_GRID_H    = BALL_RADIUS / 4.0
                BALL_START[:] = [0.0, 4.0, 0.0]
                new_mesh = pv.Sphere(radius=BALL_RADIUS, theta_resolution=22, phi_resolution=22)
            # rebuild sim internals with new shape
            sim.grid_h = float(MPM_GRID_H)
            sim._build_neighbourhood()
            sim.apply_material(sim.material_key)
            [s.reset() for s in sims]
            # recompute mesh bindings
            new_verts = np.array(new_mesh.points, dtype=float)
            if not NO_SKIN:
                _bind[0], _bind[1], _bind[2] = compute_bindings(new_verts, REST_POS)
                _rebuild_skin_arrays(_bind[0], _bind[1], _bind[2])
                iso_value_ref[0] = compute_iso_value(REST_POS, sim.rest_covs,
                                                      target_radius=BALL_RADIUS if SHAPE=="sphere" else None)
                # swap the pyvista skin mesh  -  ShallowCopy replaces VTK geometry in-place
                # so the render loop's reference to `skin` stays valid with the new mesh data
                skin.ShallowCopy(new_mesh)
                skin.points = gpu_skin()
                # Hide fluid surface; replace solid skin (topology changed so re-add needed)
                fluid_actor[0].SetVisibility(False)
                if skin_actors[0]: pl.remove_actor(skin_actors[0])
                skin_actors[0] = pl.add_mesh(skin, color=sim.mat["mesh_color"],
                                              style="surface", show_edges=False, opacity=_skin_opacity)
            drift_st["peak_mean"] = 0.0; drift_st["peak_max"] = 0.0
            part_cloud.points = sim.positions.astype(np.float32)
        elif k == "Up":       sim.mu_scale = min(sim.mu_scale * 1.2, 10.0)
        elif k == "Down":     sim.mu_scale = max(sim.mu_scale * 0.8, 0.1)
        elif k == "7":
            # Cinematic squash-shot: low, slightly to the side, looking at ground level
            pl.camera_position = [(3.5, 0.55, 5.0), (0.0, 0.25, 0.0), (0.0, 1.0, 0.0)]
            pl.render()
        elif k in ("n", "N"):
            if paused[0]:
                step_once[0] = True
            else:
                substep_mode[0] = not substep_mode[0]
                print(f"[substep] mode {'ON' if substep_mode[0] else 'OFF'}")
        elif k in ("bracketleft", "LeftBracket", "["):
            playback_speed[0] = max(0.1, playback_speed[0] * 0.5)
            print(f"[playback] {playback_speed[0]:.2f}× (lower = slower wall clock)")
        elif k in ("bracketright", "RightBracket", "]"):
            playback_speed[0] = min(4.0, playback_speed[0] * 2.0)
            print(f"[playback] {playback_speed[0]:.2f}×")
        elif k == "0":
            bookmark[0] = {
                "camera": _camera_snapshot(pl),
                "sim":    _sim_snapshot(sim),
            }
            print(f"[bookmark] saved  frame={sim.frame}  dropped={sim.dropped}")
        elif k == "9":
            if bookmark[0] is None:
                print("[bookmark] nothing saved  -  press [0] first")
            else:
                _sim_restore(sim, bookmark[0]["sim"])
                _camera_restore(pl, bookmark[0]["camera"])
                part_cloud.points = sim.positions.astype(np.float32)
                if not NO_SKIN and skin_actors[0]:
                    skin.points = gpu_skin()
                print(f"[bookmark] restored  frame={sim.frame}")
        elif k in ("f", "F"):
            shot = CAPTURE_DIR / f"frame_{sim.frame:06d}.png"
            pl.screenshot(str(shot))
            print(f"[capture] {shot}")
        elif k in ("v", "V"):
            if not recording[0]:
                out = Path(RECORD_PATH) if RECORD_PATH else (
                    CAPTURE_DIR / f"recording_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
                out.parent.mkdir(parents=True, exist_ok=True)
                pl.open_movie(str(out), framerate=TARGET_FPS, quality=8)
                recording[0] = True
                record_file[0] = str(out)
                print(f"[record] started → {out}")
            else:
                if getattr(pl, "mwriter", None) is not None:
                    pl.mwriter.close()
                    pl.mwriter = None
                recording[0] = False
                print(f"[record] stopped → {record_file[0]}")
                record_file[0] = None
        elif k in ("t", "T"):
            # Toggle tear-mode anchors.  When on, top 20% and bottom 20% of
            # particles (by rest y) are pulled apart in opposite y directions.
            new_state = 0 if _ti_tear_active[None] == 1 else 1
            _ti_tear_active[None] = new_state
            print(f"[tear] {'ON' if new_state else 'OFF'}")
        elif k in MATERIALS:
            sim.apply_material(k)
            drift_st["peak_mean"] = 0.0
            drift_st["peak_max"]  = 0.0
            if MATERIALS[k]["model"] == "fluid":
                # Fluid/Gas: hide mesh, show particles as spheres
                if skin_actors[0]: skin_actors[0].SetVisibility(False)
                fluid_actor[0].SetVisibility(False)
                part_actor.GetProperty().SetOpacity(1.0)
                col = MATERIALS[k]["mesh_color"]
                r, g, b = int(col[1:3],16)/255, int(col[3:5],16)/255, int(col[5:7],16)/255
                part_actor.GetProperty().SetColor(r, g, b)
            else:
                # Switching TO solid: by default hide particles + show skin.
                # With --no-skin: keep particles visible and skip the skin entirely.
                part_actor.GetProperty().SetOpacity(_particle_opacity)
                fluid_actor[0].SetVisibility(False)
                if not NO_SKIN:
                    if skin_actors[0]: pl.remove_actor(skin_actors[0])
                    skin_actors[0] = pl.add_mesh(skin, color=MATERIALS[k]["mesh_color"],
                                                  style="surface", show_edges=False, opacity=_skin_opacity)
                    iso_value_ref[0] = compute_iso_value(REST_POS, sim.rest_covs,
                                                          target_radius=BALL_RADIUS if SHAPE=="sphere" else None)
        redraw_text()

    vtk_iren      = pl.iren.interactor
    running       = [True]
    iso_value_ref = [iso_value]
    vtk_iren.AddObserver("KeyPressEvent", _on_key)

    if RECORD_PATH:
        out = Path(RECORD_PATH)
        out.parent.mkdir(parents=True, exist_ok=True)
        pl.open_movie(str(out), framerate=TARGET_FPS, quality=8)
        recording[0] = True
        record_file[0] = str(out)
        print(f"[record] auto-started → {out}")

    # --- main render loop ---
    frame_dt = 1.0 / TARGET_FPS
    frame    = 0

    while running[0]:
        ft = time.perf_counter()

        # ── Xbox controller polling ───────────────────────────────────────────
        if joy:
            pygame.event.pump()
            def _ax(i):   # read axis, dead-zone 0.12
                v = joy.get_axis(i)
                return v if abs(v) > 0.12 else 0.0

            # Xbox One For Windows  -  pygame DirectInput mapping:
            # Axis 0=LSX, 1=LSY, 2=RSX, 3=RSY, 4=LT(-1..1), 5=RT(-1..1)
            lx  =  _ax(0)
            lz  =  _ax(1)
            rt  = (_ax(5) + 1.0) / 2.0   # -1 at rest → 0, +1 full → 1
            lt  = (_ax(4) + 1.0) / 2.0

            # Hammer is a floating tool  -  no gravity.
            # Left stick: horizontal.  RT: swing (snap to speed instantly).  LT: lift.
            STICK_FORCE  = 0.65
            SWING_SPEED  = 0.22   # m/frame (~6.6 m/s)  -  committed downward snap
            LIFT_SPEED   = 0.12   # m/frame (~3.6 m/s)
            DAMPING      = 0.87

            _hvx[0] = _hvx[0] * DAMPING + lx * STICK_FORCE * frame_dt
            _hvz[0] = _hvz[0] * DAMPING + lz * STICK_FORCE * frame_dt

            if rt > 0.1:
                # Snap to swing speed  -  no slow build-up, feels like a real strike
                _hvy[0] = -SWING_SPEED * rt
            elif lt > 0.1:
                _hvy[0] = LIFT_SPEED * lt
            else:
                _hvy[0] *= DAMPING   # coast to a stop when no input

            nx = float(np.clip(_hx[0] + _hvx[0], -4.0, 4.0))
            nz = float(np.clip(_hz[0] + _hvz[0], -4.0, 4.0))
            ny = float(np.clip(_hy[0] + _hvy[0], CONTACT_ZONE + HAMMER_RADIUS * 0.3, 9.0))

            # Stop vertical velocity at floor/ceiling
            if ny <= CONTACT_ZONE + HAMMER_RADIUS * 0.3 or ny >= 9.0:
                _hvy[0] = 0.0

            _move_hammer(nx, ny, nz, _hvx[0], _hvy[0], _hvz[0])

            # A button → drop ball,  B → reset,  X/Y/LB/RB → material
            if joy.get_button(0):  [s.drop() for s in sims]
            if joy.get_button(1):
                [s.reset() for s in sims]
                drift_st["peak_mean"] = 0.0
                drift_st["peak_max"]  = 0.0
            for btn, key in [(2,"1"),(3,"2"),(4,"3"),(5,"4")]:
                if joy.get_button(btn):
                    sim.apply_material(key)

        try:
            vtk_iren.ProcessEvents()
        except Exception:
            break

        if not paused[0] or step_once[0]:
            if substep_mode[0] or step_once[0]:
                sim.substep_once()
                step_once[0] = False
            elif SCENE == "collide" and all(s.dropped for s in sims):
                # Collide mode: shared-grid substep so bodies interact.
                # All entities scatter to the same grid before update.
                dt_sub = DT / SUBSTEPS
                _t0 = time.perf_counter()
                mat_model = sim.mat["model"]
                model_int = 2 if mat_model == "sand" else (1 if mat_model == "fluid" else 0)
                entity_descs = [
                    {"start": s.start, "end": s.end,
                     "kappa": float(s.mat["kappa"]), "mu_scale": s.mu_scale,
                     "model": model_int, "floor_damp": s.floor_damp,
                     "friction": s.friction}
                    for s in sims
                ]
                for _ in range(SUBSTEPS):
                    solver_ul.substep_multi(entity_descs, dt_sub)
                for s in sims:
                    s._finish_frame()
                if sim.frame <= 10:
                    _t_step = (time.perf_counter() - _t0) * 1000
                    print(f"[TIMING f{frame:3d}]  step={_t_step:.1f}ms  "
                          f"(~{_t_step/SUBSTEPS:.2f}ms/substep × {SUBSTEPS} substeps)")
            elif sim.dropped and sim.frame <= 10:
                _t0 = time.perf_counter()
                for s in sims: s.step()
                _t_step = (time.perf_counter() - _t0) * 1000
                print(f"[TIMING f{frame:3d}]  step={_t_step:.1f}ms  "
                      f"(~{_t_step/SUBSTEPS:.2f}ms/substep × {SUBSTEPS} substeps)")
            else:
                for s in sims: s.step()

        # --- update visuals ---
        is_fluid = sim.mat["model"] == "fluid"
        # Gather positions from ALL entities for the particle cloud.
        if len(sims) > 1:
            all_pos = np.concatenate([s.positions for s in sims], axis=0).astype(np.float32)
            all_sm  = np.concatenate([s.diag_sigma_min for s in sims], axis=0).astype(np.float32)
        else:
            all_pos = sim.positions.astype(np.float32)
            all_sm  = sim.diag_sigma_min.astype(np.float32)
        part_cloud.points = all_pos
        # Thermo view takes priority: colour by live per-particle temperature so
        # the rising melt front is visible (inferno: dark=cold → bright=hot).
        if _thermo_view:
            _temp_now = _ti_temp.to_numpy()[sim.start:sim.end].astype(np.float32)
            part_cloud.point_data["temp"] = _temp_now
        # When fracture is active, repurpose the sigma_min colour channel to show
        # damage instead: 1 - damage → 0 = fully damaged = red, 1 = intact = green.
        elif sim.fracture_active:
            dmg = _ti_damage.to_numpy()[sim.start:sim.end].astype(np.float32)
            part_cloud.point_data["sigma_min"] = (1.0 - dmg)
        else:
            part_cloud.point_data["sigma_min"] = all_sm

        # Fracture-aware skin rebind: when bonds break, particles partition
        # into connected components.  Each mesh vertex follows its dominant
        # component and zeroes weights for particles on the other side of a
        # tear, so the mesh visibly splits along the crack.  Rebind only
        # when the bond-intact count changes (cheap GPU->CPU sum once per
        # frame).  Handles BOTH directions: bonds breaking (rebind to split)
        # and reset/heal (restore original full-blend weights).
        if not NO_SKIN and sim.fracture_active and not is_fluid:
            slice_n_bonds = (sim.end - sim.start) * K_PHYS
            cur_intact = int(_ti_bond_intact.to_numpy()[sim.start:sim.end].sum())
            if cur_intact != _frac_last_intact_count:
                if cur_intact >= slice_n_bonds:
                    # All bonds intact (probably post-reset)  -  restore original weights
                    _bind[1] = _frac_orig_weights.copy()
                else:
                    comps = fracture_compute_components(sim.start, sim.end)
                    _bind[1] = fracture_rebind_skin(_frac_orig_idxs,
                                                    _frac_orig_weights, comps)
                _rebuild_skin_arrays(_bind[0], _bind[1], _bind[2])
                _frac_last_intact_count = cur_intact

        # All materials  -  GPU skinning (positions) + geometric normals.
        # When --no-skin is set the mesh is decoupled from the render entirely
        # and we skip the per-frame skin compute (no work, no actor refresh).
        if not NO_SKIN:
            skin.points = gpu_skin()
            skin.compute_normals(cell_normals=False, point_normals=True, inplace=True)

        # Drift measurement (disabled by default; also disabled without skin
        # since the metric is mesh-skin vs Gaussian iso-surface)
        if DRIFT_ENABLED and not NO_SKIN and sim.dropped and frame % 10 == 0:
            dm, dx = measure_drift(
                np.array(skin.points, dtype=float),
                sim.positions, sim.covs, iso_value_ref[0]
            )
            if dm is not None:
                drift_st["mean"] = dm
                drift_st["max"]  = dx
                if dm > drift_st["peak_mean"]:
                    drift_st["peak_mean"] = dm
                    drift_st["peak_max"]  = dx
                uyy = sim.Us[:, 1, 1]
                sm  = sim.diag_sigma_min
                print(f"[f{frame:5d}]  "
                      f"U_yy [{uyy.min():.3f},{uyy.max():.3f}]  "
                      f"spread={uyy.max()-uyy.min():.3f}  "
                      f"sigma_min min={sm.min():.3f} mean={sm.mean():.3f} "
                      f"clamped={(sm < 0.2).sum()}  "
                      f"drift mean={dm:.4f}  PEAK={drift_st['peak_mean']:.4f}")

        # Animate gripper  -  descends at 0.5 m/s until it hits the ball
        if _gripper_active[0]:
            _gripper_y_vis[0] = max(CONTACT_ZONE + 0.05, _gripper_y_vis[0] - 0.5 * frame_dt)
            _ti_gripper_y[None] = _gripper_y_vis[0]
            gripper_actor.SetPosition(0, _gripper_y_vis[0] + 0.075, 0)

        # Webcam hand tracker  -  poll latest landmarks, upload to Taichi fields,
        # update line/joint visualisation.  With hysteresis: hand stays
        # visible for HAND_HOLD_FRAMES after MediaPipe loses detection, so
        # 1-2 frame dropouts don't cause the hand to pop in/out.  EMA
        # smoothing on landmark positions when actively tracked eats jitter.
        if HAND_ENABLED and hand_tracker is not None:
            snap = hand_tracker.get_latest()
            cur_hand   = _prev_hand_world.copy()       # default: hold last good positions
            cur_active = np.zeros(N_HANDS_MAX, dtype=np.int32)
            detected_now = [False] * N_HANDS_MAX

            for h, hand in enumerate(snap["hands"][:N_HANDS_MAX]):
                detected_now[h] = True
                world_pts = image_to_world(hand["landmarks"]).astype(np.float32)
                base = h * N_HAND_PTS_PER
                if _hand_was_detected[h]:
                    # Smooth: blend new measurement with previous (EMA on landmarks)
                    prev = _prev_hand_world[base:base + N_HAND_PTS_PER]
                    cur_hand[base:base + N_HAND_PTS_PER] = (
                        (1.0 - HAND_EMA) * world_pts + HAND_EMA * prev
                    )
                else:
                    # Just re-acquired after a gap  -  snap to new position, no smoothing
                    cur_hand[base:base + N_HAND_PTS_PER] = world_pts

            # Hysteresis: increment age on missing hands; park only after sustained loss
            for h in range(N_HANDS_MAX):
                if detected_now[h]:
                    _hand_lost_age[h] = 0
                    cur_active[h]     = 1
                else:
                    _hand_lost_age[h] += 1
                    if _hand_lost_age[h] <= HAND_HOLD_FRAMES:
                        cur_active[h] = 1   # within grace period  -  keep contact live
                    else:
                        cur_active[h] = 0
                        base = h * N_HAND_PTS_PER
                        cur_hand[base:base + N_HAND_PTS_PER, 0] = 0.0
                        cur_hand[base:base + N_HAND_PTS_PER, 1] = 9999.0
                        cur_hand[base:base + N_HAND_PTS_PER, 2] = 0.0

            # Finite-difference velocity, zeroed for one frame on re-acquisition
            # so the teleport-from-park doesn't manifest as a huge velocity spike.
            inv_dt = 1.0 / max(frame_dt, 1.0/120.0)
            vel    = np.clip((cur_hand - _prev_hand_world) * inv_dt, -10.0, 10.0).astype(np.float32)
            for h in range(N_HANDS_MAX):
                if detected_now[h] and not _hand_was_detected[h]:
                    base = h * N_HAND_PTS_PER
                    vel[base:base + N_HAND_PTS_PER] = 0.0

            _ti_hand_pts.from_numpy(cur_hand)
            _ti_hand_pts_v.from_numpy(vel)
            _ti_hand_active.from_numpy(cur_active)
            _prev_hand_world = cur_hand
            for h in range(N_HANDS_MAX):
                _hand_was_detected[h] = detected_now[h]
                base = h * N_HAND_PTS_PER
                pts_slice = cur_hand[base:base + N_HAND_PTS_PER].astype(np.float32)
                hand_line_meshes[h].points  = pts_slice
                hand_joint_clouds[h].points = pts_slice

        # Refresh the HUD text every 15 frames
        if frame % 15 == 0:
            redraw_text()

        # High-contrast frame counter remains visible regardless of the HUD toggle.
        # add_text with the same `name` swaps the actor in place each frame.
        pl.add_text(f"frame {sim.frame}", position=(500, 660), color="blue",
                    font_size=28, name="frame_counter")

        # 3DGS rasterized view  -  GPU-direct (no CPU bounce).
        if gs_renderer is not None and frame % 1 == 0:
            try:
                # Use Taichi→torch zero-copy GPU tensors when possible.
                # For multi-entity (collide scene), concatenate on GPU.
                import torch as _torch
                if len(sims) > 1:
                    _gs_pos = _torch.cat([s.positions_torch() for s in sims], dim=0)
                    _gs_cov = _torch.cat([s.covs_torch()      for s in sims], dim=0)
                else:
                    _gs_pos = sim.positions_torch()
                    _gs_cov = sim.covs_torch()
                # Camera: prefer PyVista's if available, else default.
                if pl is not None:
                    _cam = pl.camera
                    _eye = np.asarray(_cam.position,    dtype=np.float32)
                    _tgt = np.asarray(_cam.focal_point, dtype=np.float32)
                    _up  = np.asarray(_cam.up,          dtype=np.float32)
                    _fov = float(_cam.view_angle)
                else:
                    _eye = np.array([0.0, 3.0, 8.0], dtype=np.float32)
                    _tgt = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                    _up  = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                    _fov = 45.0
                _img = gs_renderer.render(
                    positions = _gs_pos, covs = _gs_cov,
                    eye = _eye, target = _tgt, up = _up,
                    fov_y_deg = _fov, width = 800, height = 800,
                )
                import cv2 as _cv2
                _bgr = _cv2.cvtColor(_img, _cv2.COLOR_RGB2BGR)
                if video_writer is not None:
                    video_writer.write(_bgr)
                else:
                    _cv2.imshow("3DGS substrate", _bgr)
                    _cv2.waitKey(1)
            except Exception as _e:
                print(f"[gs-render] frame {frame} failed: {_e}")

        try:
            pl.render()   # draw everything to the screen
        except Exception:
            break

        if recording[0]:
            pl.write_frame()

        frame += 1
        # Wall-clock throttle: lower playback_speed → longer sleep → slow-mo viewing
        rem = (frame_dt / max(playback_speed[0], 0.05)) - (time.perf_counter() - ft)
        if rem > 0:
            time.sleep(rem)

    if recording[0] and getattr(pl, "mwriter", None) is not None:
        pl.mwriter.close()
        pl.mwriter = None
        print(f"[record] saved → {record_file[0]}")

    print(f"\nRan {frame} frames.")
    print(f"Peak drift  mean={drift_st['peak_mean']:.4f}  max={drift_st['peak_max']:.4f}")
    if hand_tracker is not None:
        hand_tracker.stop()


# ============================================================
# COMPARE SCENE: controlled TL and UL visualization
# ============================================================

def run_compare():
    """Drop two identical rubber spheres side by side, one running TL-APIC and
    one running UL-MLS-MPM.  Same N=120, same material, same drop height.
    Log per-(entity, frame) drift + diagnostics to CSV for plotting.

    Visualisation is intentionally minimal  -  the headline artefact for the
    paper is the CSV.  A PyVista window opens with both balls so the user
    can sanity-check the qualitative behaviour, but every keyboard control
    from run() is intentionally omitted (no hammer, no shape swap, no
    material toggle  -  those would invalidate the comparison)."""
    if SCENE not in ("compare", "collide"):
        raise ValueError("run_compare requires --scene=compare or --scene=collide")
    if N_ENTITIES != 2:
        raise ValueError("run_compare requires exactly two entities")

    # Two entities, both rubber, identical config except for solver
    sims = []
    for ent_id, (slv, spawn) in enumerate(zip(ENTITY_SOLVERS, SPAWN_POSITIONS)):
        start = ent_id * N_PER_ENTITY
        end   = start + N_PER_ENTITY
        s = GaussianBallSim(
            material_key=(_CLI.material or "1"),  # rubber by default
            entity_id=ent_id,
            start=start, end=end,
            solver_name=slv,
            spawn_position=spawn,
            grid_h=_PER_ENTITY_GRIDH[ent_id],
        )
        sims.append(s)

    # Per-entity mesh skin (CPU path  -  independent for each entity, no shared GPU state)
    skins = []
    for s in sims:
        rest_slice = REST_POS[s.start:s.end]
        # Build sphere mesh in entity-LOCAL coords; renderer offsets by spawn_position
        sphere = pv.Sphere(radius=BALL_RADIUS, theta_resolution=22, phi_resolution=22)
        verts_rest = np.array(sphere.points, dtype=float)
        iso_val    = compute_iso_value(rest_slice, s.rest_covs, target_radius=BALL_RADIUS)
        idxs, weights, offsets = compute_bindings(verts_rest, rest_slice)
        skins.append({
            "mesh":        sphere,
            "verts_rest":  verts_rest,
            "idxs":        idxs,
            "weights":     weights,
            "offsets":     offsets,
            "iso_val":     iso_val,
            "spawn":       s.spawn_position,
        })

    # ---- Polished PyVista scene (same lab look as run()) ----------------
    plotter = None
    skin_actors = [None, None]
    hud_actor   = [None]
    paused      = [False]
    hud_on      = [True]
    running     = [True]
    if not HEADLESS:
        pv.global_theme.allow_empty_mesh = True
        plotter = pv.Plotter(window_size=(1280, 720),
                             title=f"GaussianFlesh TL-APIC  vs  PhysGaussian UL-MLS-MPM  (N={N_PER_ENTITY})")
        plotter.set_background("#f0f2f5")

        # Ground, grid lines, back wall  -  copied from run() for visual parity
        ground = pv.Plane(center=(0, 0, 0), direction=(0, 1, 0), i_size=14, j_size=14)
        plotter.add_mesh(ground, color="#e8eaed", opacity=1.0, show_edges=False)
        for i in np.arange(-6, 7, 1.0):
            plotter.add_mesh(pv.Line((i, 0.001, -6), (i, 0.001, 6)),
                             color="#d0d4da", line_width=1)
            plotter.add_mesh(pv.Line((-6, 0.001, i), (6, 0.001, i)),
                             color="#d0d4da", line_width=1)
        back_wall = pv.Plane(center=(0, 4, -6), direction=(0, 0, 1), i_size=14, j_size=10)
        plotter.add_mesh(back_wall, color="#edf0f4", opacity=1.0)

        plotter.remove_all_lights()
        plotter.add_light(pv.Light(position=(4, 10, 6),  focal_point=(0, 0, 0), intensity=0.8))
        plotter.add_light(pv.Light(position=(-4, 8, -2), focal_point=(0, 0, 0), intensity=0.5))
        plotter.add_light(pv.Light(position=(0,  2, 8),  focal_point=(0, 0, 0), intensity=0.3))

        # Skinned meshes: red denotes TL and blue denotes the UL baseline.
        # When --no-skin is set, hide the mesh and reveal each entity's particle cloud.
        ent_colors  = ["#e63946", "#457b9d"]
        ent_labels  = ["TL-APIC (ours)", "UL-MLS-MPM (baseline)"]
        skin_opacity = 0.0 if NO_SKIN else 1.0
        for i, (s, sk, c) in enumerate(zip(sims, skins, ent_colors)):
            sk["mesh"].points = (sk["verts_rest"] + sk["spawn"]).astype(np.float32)
            skin_actors[i] = plotter.add_mesh(sk["mesh"], color=c, style="surface",
                                              smooth_shading=True, opacity=skin_opacity)
        # Per-entity particle clouds  -  invisible by default, visible with --no-skin
        part_clouds = []
        for s, c in zip(sims, ent_colors):
            pc = pv.PolyData(s.positions.astype(np.float32))
            plotter.add_mesh(pc, color=c, point_size=8,
                             render_points_as_spheres=True,
                             opacity=(1.0 if NO_SKIN else 0.0),
                             show_scalar_bar=False)
            part_clouds.append(pc)

        # Camera framed to show both balls falling
        plotter.camera_position = [(0.0, 4.5, 10.0), (0.0, 1.5, 0.0), (0.0, 1.0, 0.0)]

        # Solver labels above each ball  -  corner text since PyVista can't anchor 3D
        plotter.add_text(ent_labels[0], position=(40,   675), color=ent_colors[0],
                         font_size=12, name="lbl_left")
        plotter.add_text(ent_labels[1], position=(900,  675), color=ent_colors[1],
                         font_size=12, name="lbl_right")

        # Keyboard callbacks  -  only the ones that make sense in compare mode.
        # NO shape/material switching (would invalidate the comparison).
        def _drop_all():
            for s in sims:
                s.drop()
        def _reset_all():
            for s in sims:
                s.reset()
        def _toggle_pause():
            paused[0] = not paused[0]
        def _toggle_hud():
            hud_on[0] = not hud_on[0]
            if hud_actor[0] is not None:
                plotter.remove_actor(hud_actor[0]); hud_actor[0] = None
        def _quit():
            running[0] = False
        plotter.add_key_event("space", _drop_all)
        plotter.add_key_event("r",     _reset_all)
        plotter.add_key_event("R",     _reset_all)
        plotter.add_key_event("p",     _toggle_pause)
        plotter.add_key_event("P",     _toggle_pause)
        plotter.add_key_event("h",     _toggle_hud)
        plotter.add_key_event("H",     _toggle_hud)
        plotter.add_key_event("q",     _quit)
        plotter.add_key_event("Q",     _quit)
        plotter.add_key_event("Escape",_quit)

        plotter.show(interactive_update=True, auto_close=False)

    # ---- CSV setup -----------------------------------------------------
    csv_p = evaluate_drift.csv_path(CSV_DIR, run_tag=None)
    logger = evaluate_drift.DriftLogger(csv_p)
    print(f"[compare] N_per_entity={N_PER_ENTITY}  frames={FRAMES}")
    print(f"[compare] entity 0: {sims[0].solver_name}  spawn={sims[0].spawn_position}")
    print(f"[compare] entity 1: {sims[1].solver_name}  spawn={sims[1].spawn_position}")
    print(f"[compare] CSV -> {csv_p}")
    print(f"[compare] Controls: SPACE=drop  R=reset  P=pause  H=hud  Q=quit")

    # Auto-drop both entities so the figure starts in motion
    for s in sims:
        s.drop()

    # Running drift state for the HUD
    last_drift = [(0.0, 0.0), (0.0, 0.0)]   # [(mean, max) per entity]
    peak_drift = [0.0, 0.0]

    def _redraw_hud(frame_idx, ms):
        """One HUD block per entity + global controls reminder."""
        if not hud_on[0] or plotter is None:
            return
        if hud_actor[0] is not None:
            plotter.remove_actor(hud_actor[0])
            hud_actor[0] = None
        lines = [
            f"  TL/UL comparison | N={N_PER_ENTITY}/entity   frame {frame_idx:4d}/{FRAMES}   {ms:.1f} ms/entity",
            f"  {'─' * 70}",
            f"  TL-APIC (ours)        drift_mean={last_drift[0][0]:.4f}  drift_max={last_drift[0][1]:.4f}   peak={peak_drift[0]:.4f}",
            f"  UL-MLS-MPM (baseline) drift_mean={last_drift[1][0]:.4f}  drift_max={last_drift[1][1]:.4f}   peak={peak_drift[1]:.4f}",
            f"  {'─' * 70}",
            f"  [SPACE] drop   [R] reset   [P] pause   [H] hud   [Q] quit",
        ]
        hud_actor[0] = plotter.add_text("\n".join(lines), position="upper_left",
                                        font_size=9, color="#1a2540", name="hud")

    # ---- main loop ----------------------------------------------------
    try:
        frame = 0
        last_render = time.perf_counter()
        while running[0] and frame < FRAMES:
            if paused[0]:
                if plotter is not None:
                    plotter.update()
                time.sleep(0.02)
                continue

            t0 = time.perf_counter()
            for s in sims:
                s.step()
            frame_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(sims))

            # Per-entity drift + diagnostics + skin update
            for i, (s, sk) in enumerate(zip(sims, skins)):
                skin_verts = update_verts_local(
                    sk["idxs"], sk["weights"], sk["offsets"],
                    s.positions, s.Us,
                )
                d_mean, d_max = measure_drift(skin_verts, s.positions, s.covs, sk["iso_val"])
                if d_mean is None:
                    d_mean, d_max = 0.0, 0.0
                last_drift[i] = (d_mean, d_max)
                peak_drift[i] = max(peak_drift[i], d_max)

                stats = evaluate_drift.per_entity_diagnostics(
                    _ti_U.to_numpy(), _ti_F.to_numpy(), _ti_sm.to_numpy(),
                    s.start, s.end,
                )
                logger.log(
                    solver_type=s.solver_name,
                    entity_id=s.entity_id,
                    frame=frame,
                    wall_clock_ms=frame_ms,
                    drift_mean=d_mean,
                    drift_max=d_max,
                    **stats,
                )

                if plotter is not None:
                    sk["mesh"].points = skin_verts.astype(np.float32)
                    if NO_SKIN:
                        part_clouds[i].points = s.positions.astype(np.float32)

            # Throttle HUD redraws  -  text actor recreation is expensive
            if plotter is not None and (frame % 5 == 0):
                _redraw_hud(frame, frame_ms)

            # ~30 fps render cap so the GPU isn't fighting itself
            if plotter is not None:
                now = time.perf_counter()
                if now - last_render >= (1.0 / 30.0):
                    plotter.update()
                    last_render = now

            if frame % 200 == 0:
                print(f"  frame {frame:4d}/{FRAMES}  "
                      f"TL drift_mean={last_drift[0][0]:.4f}  "
                      f"UL drift_mean={last_drift[1][0]:.4f}  "
                      f"ms/entity={frame_ms:.1f}")
            frame += 1
    finally:
        logger.close()
        if plotter is not None:
            try: plotter.close()
            except Exception: pass
        print(f"[compare] done. CSV at {csv_p}")
        print(f"[compare] peak drift  TL={peak_drift[0]:.4f}  UL={peak_drift[1]:.4f}")


def run_gs_headless():
    """No PyVista, no diagnostics, just substep + gs_render.  Optionally
    record to video.  Designed for batch demos at full GPU speed.

    Scenario: drop at frame 0, run for FRAMES frames.  No interactive input."""
    import cv2 as _cv2
    import torch as _torch
    from gs_render import GSRenderer
    if not GS_RENDER_ENABLED:
        raise ValueError("run_gs_headless requires --gs-render or --video")

    def _project_points(points, eye, target, up, fov_y_deg, width, height):
        """Project world points to pixels using the same look-at convention as
        gs_render.  Used only for 2D guide overlays in recorded demos."""
        pts = np.asarray(points, dtype=np.float64)
        eye64 = np.asarray(eye, dtype=np.float64)
        target64 = np.asarray(target, dtype=np.float64)
        up64 = np.asarray(up, dtype=np.float64)
        fwd = target64 - eye64
        fwd /= np.linalg.norm(fwd) + 1e-12
        right = np.cross(fwd, up64)
        right /= np.linalg.norm(right) + 1e-12
        up_c = np.cross(right, fwd)
        rel = pts - eye64[None, :]
        x = rel @ right
        y = rel @ up_c
        z = rel @ fwd
        f = 0.5 * height / np.tan(np.radians(fov_y_deg) * 0.5)
        px = width * 0.5 + f * x / np.maximum(z, 1e-6)
        # gs_render uses COLMAP-style camera rows [right, -up, forward], so
        # positive world-up projects downward in image coordinates here.
        py = height * 0.5 + f * y / np.maximum(z, 1e-6)
        return np.stack([px, py], axis=1), z

    def _draw_floor_overlay(img, eye, target, up):
        """Light screen-space floor/grid reference for gs-only videos.
        This is a visual guide only; the Gaussian rasterizer has no ground
        plane primitive."""
        original = img.copy()
        h, w = img.shape[:2]
        color_grid = (220, 224, 228)
        color_axis = (185, 190, 198)
        horizon = int(h * 0.63)
        bottom = int(h * 0.96)
        left_top = int(w * 0.25)
        right_top = int(w * 0.75)
        left_bot = int(w * -0.05)
        right_bot = int(w * 1.05)
        # Horizontal perspective lines.
        for i in range(8):
            t = i / 7.0
            y = int(horizon + (bottom - horizon) * (t ** 1.7))
            xl = int(left_top + (left_bot - left_top) * t)
            xr = int(right_top + (right_bot - right_top) * t)
            _cv2.line(img, (xl, y), (xr, y), color_axis if i == 0 else color_grid, 1, _cv2.LINE_AA)
        # Vanishing grid lines.
        for i in range(9):
            t = i / 8.0
            x_top = int(left_top + (right_top - left_top) * t)
            x_bot = int(left_bot + (right_bot - left_bot) * t)
            _cv2.line(img, (x_top, horizon), (x_bot, bottom), color_grid, 1, _cv2.LINE_AA)
        obj_mask = np.any(original < 245, axis=2)
        img[obj_mask] = original[obj_mask]
        return img

    def _draw_gripper_overlay(img, eye, target, up, gy, x_half, z_half):
        """Draw the scripted squash plate in screen space."""
        if gy > 1000.0:
            return img
        h, w = img.shape[:2]
        y = float(gy)
        corners = np.array([
            [-x_half, y, -z_half],
            [ x_half, y, -z_half],
            [ x_half, y,  z_half],
            [-x_half, y,  z_half],
        ], dtype=np.float32)
        pts, depth = _project_points(corners, eye, target, up, 45.0, w, h)
        if np.any(depth <= 0.0):
            return img
        poly = np.round(pts).astype(np.int32)
        overlay = img.copy()
        _cv2.fillConvexPoly(overlay, poly, (184, 188, 194), lineType=_cv2.LINE_AA)
        _cv2.polylines(overlay, [poly], True, (95, 100, 110), 2, _cv2.LINE_AA)
        return _cv2.addWeighted(overlay, 0.38, img, 0.62, 0.0)

    def _draw_side_plate_overlay(img, eye, target, up, x, y_min, y_max, z_half):
        """Draw one vertical side plate in screen space."""
        if abs(x) > 1000.0:
            return img
        h, w = img.shape[:2]
        corners = np.array([
            [x, y_min, -z_half],
            [x, y_max, -z_half],
            [x, y_max,  z_half],
            [x, y_min,  z_half],
        ], dtype=np.float32)
        pts, depth = _project_points(corners, eye, target, up, 45.0, w, h)
        if np.any(depth <= 0.0):
            return img
        poly = np.round(pts).astype(np.int32)
        overlay = img.copy()
        _cv2.fillConvexPoly(overlay, poly, (184, 188, 194), lineType=_cv2.LINE_AA)
        _cv2.polylines(overlay, [poly], True, (95, 100, 110), 2, _cv2.LINE_AA)
        return _cv2.addWeighted(overlay, 0.42, img, 0.58, 0.0)

    def _draw_hammer_overlay(img, eye, target, up, hpos):
        """Draw the scripted drag sphere in screen space."""
        if hpos[1] > 1000.0:
            return img
        h, w = img.shape[:2]
        pts, depth = _project_points(np.array([hpos], dtype=np.float32),
                                     eye, target, up, 45.0, w, h)
        if depth[0] <= 0.0:
            return img
        px, py = np.round(pts[0]).astype(np.int32)
        f = 0.5 * h / np.tan(np.deg2rad(45.0) * 0.5)
        rr = int(np.clip(f * HAMMER_RADIUS / depth[0], 8, 46))
        overlay = img.copy()
        _cv2.circle(overlay, (px, py), rr, (64, 72, 84), -1, _cv2.LINE_AA)
        _cv2.circle(overlay, (px, py), rr, (20, 24, 32), 2, _cv2.LINE_AA)
        return _cv2.addWeighted(overlay, 0.70, img, 0.30, 0.0)

    # Sim setup
    sim = GaussianBallSim(material_key=(_CLI.material or "1"))
    sim.drop()

    if _CLI.start_bottom is not None:
        sl = slice(sim.start, sim.end)
        cur_pos = _ti_pos.to_numpy()
        min_y = float(cur_pos[sl, 1].min())
        requested_gap = float(_CLI.start_bottom) - float(_CLI.floor_coordinate)
        target_min_y = float(CONTACT_ZONE + requested_gap)
        cur_pos[sl, 1] += target_min_y - min_y
        _ti_pos.from_numpy(cur_pos)
        cur_vel = _ti_vel.to_numpy()
        cur_vel[sl] = 0.0
        _ti_vel.from_numpy(cur_vel)
        sim.positions = cur_pos[sl].astype(float)
        sim.velocities = cur_vel[sl].astype(float)
        print(f"[gs-headless] requested source coordinates: bottom={_CLI.start_bottom:.3f}, "
              f"floor={_CLI.floor_coordinate:.3f}, gap={requested_gap:.3f}; "
              f"internal min_y={target_min_y:.3f}")

    if THERMO_ENABLED and not SQUASH_RELEASE and not SIDE_SQUASH_RELEASE and not DUAL_SQUASH_RELEASE:
        sl = slice(sim.start, sim.end)
        cur_pos = _ti_pos.to_numpy()
        min_y = float(cur_pos[sl, 1].min())
        target_min_y = float(CONTACT_ZONE + 0.03)
        cur_pos[sl, 1] += target_min_y - min_y
        _ti_pos.from_numpy(cur_pos)
        cur_vel = _ti_vel.to_numpy()
        cur_vel[sl] = 0.0
        _ti_vel.from_numpy(cur_vel)
        sim.positions = cur_pos[sl].astype(float)
        sim.velocities = cur_vel[sl].astype(float)
        print(f"[gs-headless] thermo start: min_y={target_min_y:.2f}, "
              f"hot_floor={HOT_FLOOR_TEMP}")

    # --- Multi-material split (--ply-split MATKEY:FRACTION) ---------------
    # Low-side FRACTION of the body, measured along --ply-split-axis, is
    # overwritten with the requested material via the full per-particle path
    # (_apply_material_to_mask).  The default y-axis preserves the original
    # bottom/top split; x-axis gives a clearer left/right demo for pillows.
    _split_local_mask = None
    _split_mat_key = None
    if PLY_SPLIT and ":" in PLY_SPLIT:
        _mk, _frac = PLY_SPLIT.split(":")
        _frac = float(_frac)
        if _mk in MATERIALS:
            _axis_idx = {"x": 0, "y": 1, "z": 2}.get(PLY_SPLIT_AXIS, 1)
            rest_axis = REST_POS[sim.start:sim.end, _axis_idx]
            a_lo, a_hi = float(rest_axis.min()), float(rest_axis.max())
            thresh = a_lo + _frac * (a_hi - a_lo)
            local_bottom = rest_axis < thresh
            _split_local_mask = local_bottom.copy()
            _split_mat_key = _mk
            abs_mask = np.zeros(N_PARTICLES, dtype=bool)
            abs_mask[sim.start:sim.end] = local_bottom
            _apply_material_to_mask(MATERIALS[_mk], abs_mask)
            print(f"[gs-headless] split: low-{PLY_SPLIT_AXIS} {int(local_bottom.sum())}/"
                  f"{sim.end-sim.start} particles -> {MATERIALS[_mk]['name']} "
                  f"(rest -> {sim.mat['name']})")

            if (not SQUASH_RELEASE and not SIDE_SQUASH_RELEASE
                    and not DUAL_SQUASH_RELEASE and not DRAG_DEMO):
                # Poster collision setup: start just above the floor with a
                # downward velocity so the recorded clip shows impact rather
                # than several seconds of empty fall.
                sl = slice(sim.start, sim.end)
                cur_pos = _ti_pos.to_numpy()
                min_y = float(cur_pos[sl, 1].min())
                target_min_y = float(CONTACT_ZONE + 0.85)
                cur_pos[sl, 1] += target_min_y - min_y
                _ti_pos.from_numpy(cur_pos)
                cur_vel = _ti_vel.to_numpy()
                cur_vel[sl, 1] = IMPACT_VY
                _ti_vel.from_numpy(cur_vel)
                sim.positions = cur_pos[sl].astype(float)
                sim.velocities = cur_vel[sl].astype(float)
                print(f"[gs-headless] collision start: min_y={target_min_y:.2f}, "
                      f"initial vy={IMPACT_VY:.1f} m/s")

    if SQUASH_RELEASE or SIDE_SQUASH_RELEASE or DUAL_SQUASH_RELEASE or DRAG_DEMO:
        sl = slice(sim.start, sim.end)
        cur_pos = _ti_pos.to_numpy()
        min_y = float(cur_pos[sl, 1].min())
        target_min_y = float(CONTACT_ZONE + 0.03)
        cur_pos[sl, 1] += target_min_y - min_y
        _ti_pos.from_numpy(cur_pos)
        cur_vel = _ti_vel.to_numpy()
        cur_vel[sl] = 0.0
        _ti_vel.from_numpy(cur_vel)
        sim.positions = cur_pos[sl].astype(float)
        sim.velocities = cur_vel[sl].astype(float)
        _ti_gripper_y[None] = 9999.0
        _ti_bottom_gripper_y[None] = -9999.0
        _ti_side_plate_lx[None] = -9999.0
        _ti_side_plate_rx[None] = 9999.0
        _ti_hammer_pos[None] = ti.Vector([0.0, 9999.0, 0.0])
        _ti_hammer_vel[None] = ti.Vector([0.0, 0.0, 0.0])
        print(f"[gs-headless] squash start: min_y={target_min_y:.2f}, "
              f"target ratio={SQUASH_RATIO:.2f}")

    # Legacy --multi-material (top half) path
    _multi = getattr(_CLI, "multi_material", None)
    if _multi and _multi in MATERIALS:
        mat_top = MATERIALS[_multi]
        rest_y = _ti_rest.to_numpy()[sim.start:sim.end, 1]
        top_mask = rest_y > float(rest_y.mean())
        sl = slice(sim.start, sim.end)
        cm  = _ti_mu.to_numpy();  cm[sl][top_mask]  = float(mat_top["mu"]);  _ti_mu.from_numpy(cm)
        cl  = _ti_lam.to_numpy(); cl[sl][top_mask] = float(mat_top["lam"]); _ti_lam.from_numpy(cl)

    # GS appearance
    if SHAPE == "ply":
        _opac = _PLY_OPACITY_PER_ENTITY[0]
        _shs  = _PLY_SHS_PER_ENTITY[0]
    else:
        _hex = sim.mat.get("mesh_color", "#888888").lstrip("#")
        _rgb = np.array([int(_hex[i:i+2], 16) / 255.0 for i in (0, 2, 4)])
        _sh_c0 = 0.28209479
        _sh_c1 = 0.48860251
        _opac = np.full((N_PARTICLES, 1), 0.85, dtype=np.float32)
        _shs  = np.zeros((N_PARTICLES, 16, 3), dtype=np.float32)
        _rgb_all = np.tile(_rgb[None, :], (N_PARTICLES, 1)).astype(np.float32)
        if _split_local_mask is not None and _split_mat_key in MATERIALS:
            _split_hex = MATERIALS[_split_mat_key].get("mesh_color", "#888888").lstrip("#")
            _split_rgb = np.array([int(_split_hex[i:i+2], 16) / 255.0 for i in (0, 2, 4)],
                                  dtype=np.float32)
            _rgb_all[sim.start:sim.end][_split_local_mask] = _split_rgb
        _shs[:, 0, :] = ((_rgb_all - 0.5) / _sh_c0).astype(np.float32)
        _shs[:, 2, :] = (0.5 / _sh_c1)
    gs = GSRenderer(_opac, _shs, bg=(1.0, 1.0, 1.0))

    # Video writer or display
    if VIDEO_OUT:
        fourcc = _cv2.VideoWriter_fourcc(*"mp4v")
        writer = _cv2.VideoWriter(VIDEO_OUT, fourcc, 30.0, (800, 800))
        print(f"[gs-headless] recording to {VIDEO_OUT}")
    else:
        writer = None
        _cv2.namedWindow("3DGS substrate", _cv2.WINDOW_NORMAL)
        _cv2.resizeWindow("3DGS substrate", 800, 800)

    # Auto-frame camera.  For a collision shot (PLY_SPLIT set), bias toward the
    # floor/impact region instead of the full drop; otherwise the ficus is too
    # small to show deformation or material contrast.
    _p0 = sim.positions.astype(np.float64)
    _ctr = _p0.mean(axis=0)
    _rad = float(np.linalg.norm(_p0 - _ctr, axis=1).max()) + 1e-6
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if PLY_SPLIT or SQUASH_RELEASE or SIDE_SQUASH_RELEASE or DUAL_SQUASH_RELEASE or THERMO_ENABLED:
        # Keep the falling object in frame while still making the floor visible.
        # This is a middle ground between the earlier full-drop shot (too tiny)
        # and the tight impact shot (clipped the ficus before impact).
        _shot_y = max(float(CONTACT_ZONE) + 2.0 * _rad, 1.8)
        target = np.array([_ctr[0], _shot_y, _ctr[2]], dtype=np.float32)
        _cam_dist = max(_rad * 5.4, 4.8)
        eye = np.array([_ctr[0] + 0.30 * _cam_dist,
                        _shot_y + 0.14 * _cam_dist,
                        _ctr[2] + 0.95 * _cam_dist], dtype=np.float32)
    else:
        target = np.array([_ctr[0], _ctr[1], _ctr[2]], dtype=np.float32)
        _cam_dist = _rad * 2.4
        eye = np.array([_ctr[0] + 0.35 * _cam_dist,
                        _ctr[1] + 0.15 * _cam_dist,
                        _ctr[2] + 0.92 * _cam_dist], dtype=np.float32)
    print(f"[gs-headless] auto-frame: center={_ctr.round(2)} radius={_rad:.2f} "
          f"eye={eye.round(2)} target={target.round(2)}")

    # Main loop  -  no PyVista, no diagnostics
    _squash = None
    if SQUASH_RELEASE:
        _p = sim.positions.astype(np.float64)
        _y_min = float(_p[:, 1].min())
        _y_max = float(_p[:, 1].max())
        _h0 = max(_y_max - _y_min, 1e-4)
        _plate_start = _y_max + 0.35 * _h0
        _plate_low = _y_min + max(0.20, SQUASH_RATIO) * _h0
        _x_half = max(1.25 * float(np.max(np.abs(_p[:, 0]))), 1.2)
        _z_half = max(1.25 * float(np.max(np.abs(_p[:, 2]))), 1.2)
        _squash = {
            "hold0": 45,
            "press": 210,
            "hold": 120,
            "release": 20,
            "start": _plate_start,
            "low": _plate_low,
            "x_half": _x_half,
            "z_half": _z_half,
        }
        print(f"[gs-headless] squash plate: y {_plate_start:.2f} -> {_plate_low:.2f} "
              f"(h0={_h0:.2f}, xhalf={_x_half:.2f}, zhalf={_z_half:.2f})")

    _side_squash = None
    if SIDE_SQUASH_RELEASE:
        _p = sim.positions.astype(np.float64)
        _x_min = float(_p[:, 0].min())
        _x_max = float(_p[:, 0].max())
        _y_min = float(_p[:, 1].min())
        _y_max = float(_p[:, 1].max())
        _z_half = max(1.25 * float(np.max(np.abs(_p[:, 2]))), 1.2)
        _w0 = max(_x_max - _x_min, 1e-4)
        _cx = 0.5 * (_x_min + _x_max)
        _half_low = max(0.18, 0.5 * SQUASH_RATIO) * _w0
        _side_squash = {
            "hold0": 45,
            "press": 210,
            "hold": 120,
            "release": 35,
            "lx_start": _x_min - 0.35 * _w0,
            "rx_start": _x_max + 0.35 * _w0,
            "lx_low": _cx - _half_low,
            "rx_low": _cx + _half_low,
            "y_min": _y_min,
            "y_max": _y_max + 0.65 * (_y_max - _y_min),
            "z_half": _z_half,
        }
        print(f"[gs-headless] side plates: x "
              f"{_side_squash['lx_start']:.2f}/{_side_squash['rx_start']:.2f} -> "
              f"{_side_squash['lx_low']:.2f}/{_side_squash['rx_low']:.2f} "
              f"(w0={_w0:.2f}, zhalf={_z_half:.2f})")

    _dual_squash = None
    if DUAL_SQUASH_RELEASE:
        _p = sim.positions.astype(np.float64)
        _y_min = float(_p[:, 1].min())
        _y_max = float(_p[:, 1].max())
        _h0 = max(_y_max - _y_min, 1e-4)
        _cy = 0.5 * (_y_min + _y_max)
        _top_start = _y_max + 0.32 * _h0
        _bot_start = _y_min - 0.32 * _h0
        _half_low = max(0.15, 0.5 * SQUASH_RATIO) * _h0
        _x_half = max(1.25 * float(np.max(np.abs(_p[:, 0]))), 1.2)
        _z_half = max(1.25 * float(np.max(np.abs(_p[:, 2]))), 1.2)
        _dual_squash = {
            "hold0": 45,
            "press": 210,
            "hold": 120,
            "release": 35,
            "top_start": _top_start,
            "bot_start": _bot_start,
            "top_low": _cy + _half_low,
            "bot_low": _cy - _half_low,
            "x_half": _x_half,
            "z_half": _z_half,
        }
        print(f"[gs-headless] dual plates: y "
              f"{_bot_start:.2f}/{_top_start:.2f} -> "
              f"{_dual_squash['bot_low']:.2f}/{_dual_squash['top_low']:.2f} "
              f"(h0={_h0:.2f}, xhalf={_x_half:.2f}, zhalf={_z_half:.2f})")

    _drag_demo = None
    if DRAG_DEMO:
        _p = sim.positions.astype(np.float64)
        _x_min = float(_p[:, 0].min())
        _x_max = float(_p[:, 0].max())
        _y_min = float(_p[:, 1].min())
        _y_max = float(_p[:, 1].max())
        _z_ctr = float(np.median(_p[:, 2]))
        _w0 = max(_x_max - _x_min, 1e-4)
        _h0 = max(_y_max - _y_min, 1e-4)
        # Gentle lateral sweeps.  The first touches the metal pot/base low down;
        # the second touches the jelly canopy higher up.  Contact sphere moves
        # left->right then right->left so both phases are visually separated.
        _drag_demo = {
            "pot_y": _y_min + 0.28 * _h0,
            "top_y": _y_min + 0.74 * _h0,
            "z": _z_ctr,
            "x_left": _x_min - 0.10 * _w0,
            "x_right": _x_max + 0.85 * _w0,
            "hold0": 45,
            "pot_drag": 210,
            "pause": 90,
            "top_drag": 240,
            "settle": 135,
        }
        _drag_demo["total"] = (_drag_demo["hold0"] + _drag_demo["pot_drag"]
                               + _drag_demo["pause"] + _drag_demo["top_drag"]
                               + _drag_demo["settle"])
        print(f"[gs-headless] drag demo: pot y={_drag_demo['pot_y']:.2f}, "
              f"top y={_drag_demo['top_y']:.2f}, "
              f"x {_drag_demo['x_left']:.2f}->{_drag_demo['x_right']:.2f}")

    # Thermo diagnostic: pick bottom/middle/top particle by rest-y for table printout.
    _thermo_diag_idxs = None
    _THERMO_DIAG_FRAMES = {0, 100, 200, 300}
    if THERMO_ENABLED and "melt_temp" in sim.mat:
        _rest_y = REST_POS[sim.start:sim.end, 1]
        _n_ent  = sim.end - sim.start
        _bot_local = int(np.argmin(_rest_y))
        _mid_local = int(_n_ent // 2)
        _top_local = int(np.argmax(_rest_y))
        _thermo_diag_idxs = (
            sim.start + _bot_local,
            sim.start + _mid_local,
            sim.start + _top_local,
        )
        print(f"\n{'Frame':>6}  {'Layer':>6}  {'T':>8}  {'mu':>12}  {'phase':>6}")
        print("-" * 48)

    t_start = time.perf_counter()
    for frame in range(FRAMES):
        gripper_y_now = 9999.0
        bottom_gripper_y_now = -9999.0
        side_lx_now = -9999.0
        side_rx_now = 9999.0
        hammer_pos_now = np.array([0.0, 9999.0, 0.0], dtype=np.float32)
        hammer_vel_now = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if _squash is not None:
            h0 = _squash["hold0"]
            p1 = h0 + _squash["press"]
            h1 = p1 + _squash["hold"]
            r1 = h1 + _squash["release"]
            if frame < h0:
                gripper_y_now = _squash["start"]
            elif frame < p1:
                t = (frame - h0) / max(_squash["press"] - 1, 1)
                t = t * t * (3.0 - 2.0 * t)
                gripper_y_now = (1.0 - t) * _squash["start"] + t * _squash["low"]
            elif frame < h1:
                gripper_y_now = _squash["low"]
            elif frame < r1:
                t = (frame - h1) / max(_squash["release"] - 1, 1)
                t = t * t * (3.0 - 2.0 * t)
                gripper_y_now = (1.0 - t) * _squash["low"] + t * _squash["start"]
            else:
                gripper_y_now = 9999.0
            _ti_gripper_y[None] = float(gripper_y_now)

        if _side_squash is not None:
            h0 = _side_squash["hold0"]
            p1 = h0 + _side_squash["press"]
            h1 = p1 + _side_squash["hold"]
            r1 = h1 + _side_squash["release"]
            if frame < h0:
                side_lx_now = _side_squash["lx_start"]
                side_rx_now = _side_squash["rx_start"]
            elif frame < p1:
                t = (frame - h0) / max(_side_squash["press"] - 1, 1)
                t = t * t * (3.0 - 2.0 * t)
                side_lx_now = ((1.0 - t) * _side_squash["lx_start"]
                               + t * _side_squash["lx_low"])
                side_rx_now = ((1.0 - t) * _side_squash["rx_start"]
                               + t * _side_squash["rx_low"])
            elif frame < h1:
                side_lx_now = _side_squash["lx_low"]
                side_rx_now = _side_squash["rx_low"]
            elif frame < r1:
                t = (frame - h1) / max(_side_squash["release"] - 1, 1)
                t = t * t * (3.0 - 2.0 * t)
                side_lx_now = ((1.0 - t) * _side_squash["lx_low"]
                               + t * _side_squash["lx_start"])
                side_rx_now = ((1.0 - t) * _side_squash["rx_low"]
                               + t * _side_squash["rx_start"])
            else:
                side_lx_now = -9999.0
                side_rx_now = 9999.0
            _ti_side_plate_lx[None] = float(side_lx_now)
            _ti_side_plate_rx[None] = float(side_rx_now)

        if _dual_squash is not None:
            h0 = _dual_squash["hold0"]
            p1 = h0 + _dual_squash["press"]
            h1 = p1 + _dual_squash["hold"]
            r1 = h1 + _dual_squash["release"]
            if frame < h0:
                gripper_y_now = _dual_squash["top_start"]
                bottom_gripper_y_now = _dual_squash["bot_start"]
            elif frame < p1:
                t = (frame - h0) / max(_dual_squash["press"] - 1, 1)
                t = t * t * (3.0 - 2.0 * t)
                gripper_y_now = ((1.0 - t) * _dual_squash["top_start"]
                                 + t * _dual_squash["top_low"])
                bottom_gripper_y_now = ((1.0 - t) * _dual_squash["bot_start"]
                                        + t * _dual_squash["bot_low"])
            elif frame < h1:
                gripper_y_now = _dual_squash["top_low"]
                bottom_gripper_y_now = _dual_squash["bot_low"]
            elif frame < r1:
                t = (frame - h1) / max(_dual_squash["release"] - 1, 1)
                t = t * t * (3.0 - 2.0 * t)
                gripper_y_now = ((1.0 - t) * _dual_squash["top_low"]
                                 + t * _dual_squash["top_start"])
                bottom_gripper_y_now = ((1.0 - t) * _dual_squash["bot_low"]
                                        + t * _dual_squash["bot_start"])
            else:
                gripper_y_now = 9999.0
                bottom_gripper_y_now = -9999.0
            _ti_gripper_y[None] = float(gripper_y_now)
            _ti_bottom_gripper_y[None] = float(bottom_gripper_y_now)

        if _drag_demo is not None:
            h0 = _drag_demo["hold0"]
            p1 = h0 + _drag_demo["pot_drag"]
            p2 = p1 + _drag_demo["pause"]
            p3 = p2 + _drag_demo["top_drag"]
            fps = 30.0
            if frame < h0:
                hammer_pos_now = np.array([
                    _drag_demo["x_left"],
                    _drag_demo["pot_y"],
                    _drag_demo["z"],
                ], dtype=np.float32)
            elif frame < p1:
                u = (frame - h0) / max(_drag_demo["pot_drag"] - 1, 1)
                t = u * u * (3.0 - 2.0 * u)
                dtdu = 6.0 * u * (1.0 - u)
                dx = _drag_demo["x_right"] - _drag_demo["x_left"]
                hammer_pos_now = np.array([
                    _drag_demo["x_left"] + t * dx,
                    _drag_demo["pot_y"],
                    _drag_demo["z"],
                ], dtype=np.float32)
                hammer_vel_now = np.array([
                    dx * dtdu * fps / max(_drag_demo["pot_drag"] - 1, 1),
                    0.0, 0.0,
                ], dtype=np.float32)
            elif frame < p2:
                hammer_pos_now = np.array([
                    _drag_demo["x_right"],
                    _drag_demo["pot_y"],
                    _drag_demo["z"],
                ], dtype=np.float32)
            elif frame < p3:
                u = (frame - p2) / max(_drag_demo["top_drag"] - 1, 1)
                t = u * u * (3.0 - 2.0 * u)
                dtdu = 6.0 * u * (1.0 - u)
                dx = _drag_demo["x_left"] - _drag_demo["x_right"]
                hammer_pos_now = np.array([
                    _drag_demo["x_right"] + t * dx,
                    _drag_demo["top_y"],
                    _drag_demo["z"],
                ], dtype=np.float32)
                hammer_vel_now = np.array([
                    dx * dtdu * fps / max(_drag_demo["top_drag"] - 1, 1),
                    0.0, 0.0,
                ], dtype=np.float32)
            _ti_hammer_pos[None] = ti.Vector([
                float(hammer_pos_now[0]),
                float(hammer_pos_now[1]),
                float(hammer_pos_now[2]),
            ])
            _ti_hammer_vel[None] = ti.Vector([
                float(hammer_vel_now[0]),
                float(hammer_vel_now[1]),
                float(hammer_vel_now[2]),
            ])

        sim.step()
        if THERMO_ENABLED and "melt_temp" in sim.mat:
            sl = slice(sim.start, sim.end)
            temp  = _ti_temp.to_numpy()[sl].astype(np.float32)
            phase = _ti_phase.to_numpy()[sl].astype(np.float32)
            if _thermo_diag_idxs is not None and frame in _THERMO_DIAG_FRAMES:
                _mu_all = _ti_mu.to_numpy()
                _T_all  = _ti_temp.to_numpy()
                _ph_all = _ti_phase.to_numpy()
                for _label, _pidx in zip(("bottom", "middle", "top"), _thermo_diag_idxs):
                    print(f"{frame:>6}  {_label:>6}  "
                          f"{_T_all[_pidx]:>8.2f}  "
                          f"{_mu_all[_pidx]:>12.1f}  "
                          f"{'fluid' if _ph_all[_pidx] else 'solid':>6}")
            source_temp = float(HOT_FLOOR_TEMP) if HOT_FLOOR_TEMP is not None else (
                float(sim.mat["melt_temp"]) + 60.0)
            heat01 = np.clip((temp - AMBIENT_TEMP) / max(source_temp - AMBIENT_TEMP, 1e-6), 0.0, 1.0)
            cold_rgb = np.array([0.70, 0.68, 0.62], dtype=np.float32)
            hot_rgb = np.array([1.00, 0.42, 0.08], dtype=np.float32)
            molten_rgb = np.array([1.00, 0.78, 0.28], dtype=np.float32)
            rgb = cold_rgb[None, :] * (1.0 - heat01[:, None]) + hot_rgb[None, :] * heat01[:, None]
            rgb = rgb * (1.0 - 0.65 * phase[:, None]) + molten_rgb[None, :] * (0.65 * phase[:, None])
            dc = ((rgb - 0.5) / 0.28209479).astype(np.float32)
            gs.shs[sl, 0, :] = _torch.from_numpy(dc).to(gs.device)
        pos_t = sim.positions_torch()
        cov_t = sim.covs_torch()
        img = gs.render(positions=pos_t, covs=cov_t,
                        eye=eye, target=target, up=up,
                        fov_y_deg=45.0, width=800, height=800)
        if (PLY_SPLIT or SQUASH_RELEASE or SIDE_SQUASH_RELEASE
                or DUAL_SQUASH_RELEASE or DRAG_DEMO or THERMO_ENABLED):
            img = _draw_floor_overlay(img, eye, target, up)
        if _squash is not None:
            img = _draw_gripper_overlay(
                img, eye, target, up, gripper_y_now,
                _squash["x_half"], _squash["z_half"])
        if _side_squash is not None:
            img = _draw_side_plate_overlay(
                img, eye, target, up, side_lx_now,
                _side_squash["y_min"], _side_squash["y_max"],
                _side_squash["z_half"])
            img = _draw_side_plate_overlay(
                img, eye, target, up, side_rx_now,
                _side_squash["y_min"], _side_squash["y_max"],
                _side_squash["z_half"])
        if _dual_squash is not None:
            img = _draw_gripper_overlay(
                img, eye, target, up, gripper_y_now,
                _dual_squash["x_half"], _dual_squash["z_half"])
            img = _draw_gripper_overlay(
                img, eye, target, up, bottom_gripper_y_now,
                _dual_squash["x_half"], _dual_squash["z_half"])
        if _drag_demo is not None:
            img = _draw_hammer_overlay(img, eye, target, up, hammer_pos_now)
        bgr = _cv2.cvtColor(img, _cv2.COLOR_RGB2BGR)
        if writer is not None:
            writer.write(bgr)
        else:
            _cv2.imshow("3DGS substrate", bgr)
            if _cv2.waitKey(1) == ord("q"):
                break
        if frame % 50 == 0:
            elapsed = time.perf_counter() - t_start
            fps = (frame + 1) / elapsed
            thermo_msg = ""
            if THERMO_ENABLED and "melt_temp" in sim.mat:
                _by = sim.positions[:, 1]
                _height = float(_by.max() - _by.min())
                thermo_msg = (f"  molten={sim.thermo_molten}/{sim.end-sim.start}"
                              f"  T=[{sim.thermo_temp_min:.1f},{sim.thermo_temp_max:.1f}]"
                              f"  Umean={sim.thermo_U_mean:.2f}"
                              f"  H={_height:.2f}")
            print(f"[gs-headless] frame {frame}/{FRAMES}  {fps:.1f} fps{thermo_msg}")

    if writer is not None:
        writer.release()
        print(f"[gs-headless] done. video: {VIDEO_OUT}")
    else:
        _cv2.destroyAllWindows()


def _apply_material_to_mask(mat_dict, abs_mask):
    """Overwrite ALL per-particle material fields for the particles selected
    by abs_mask (a boolean array over the full N_PARTICLES range).  Used to
    paint a second material onto part of an existing body  -  per-particle
    material identity within one connected object.

    `mat_dict` is a MATERIALS entry.  Writes μ, λ, yield, plastic mode,
    constitutive law (_ti_model), damping, relax, dp_alpha, hardening, mass."""
    def _ow(field, value):
        cur = field.to_numpy(); cur[abs_mask] = value; field.from_numpy(cur)
    _ow(_ti_mu,    float(mat_dict["mu"]))
    _ow(_ti_mu0,   float(mat_dict["mu"]))
    _ow(_ti_lam,   float(mat_dict["lam"]))
    _ow(_ti_yield, float(mat_dict.get("yield_stress", 1e9)))
    _ow(_ti_relax, float(mat_dict.get("relax_time", 0.0)))
    # damping coefficient α = damping / particle_mass; approximate with mean V0
    _mean_v0 = float(np.mean(V0)) if isinstance(V0, np.ndarray) else float(V0)
    _pm = max(float(mat_dict.get("density", 1000.0)) * _mean_v0, 1e-8)
    _ow(_ti_damping, float(mat_dict.get("damping", 0.0)) / _pm)
    j2_val = mat_dict.get("plastic_j2", 0)
    _ow(_ti_plastic_j2, int(j2_val) if isinstance(j2_val, (int, float)) else (1 if j2_val else 0))
    _ow(_ti_tension_cap, float(mat_dict.get("tension_stretch_cap", 0.0)))
    _ow(_ti_sr_gamma,    float(mat_dict.get("strain_rate_gamma", 0.0)))
    _ow(_ti_hardening_xi, float(mat_dict.get("hardening_xi", 0.0)))
    phi_deg = float(mat_dict.get("friction_angle", 0.0))
    dp_a = (np.sin(np.radians(phi_deg)) / (np.sqrt(3.0) * np.cos(np.radians(phi_deg)))
            if phi_deg > 0.0 else 0.0)
    _ow(_ti_dp_alpha, dp_a)
    mstr = mat_dict.get("model", "corotated")
    mint = 2 if mstr == "sand" else (1 if mstr == "fluid" else (3 if mstr == "stvk" else 0))
    _ow(_ti_model, mint)
    base_kappa = float(mat_dict.get("kappa", 0.0)) if mint == 1 else 0.0
    base_phase = 1 if mint == 1 else 0
    base_U = float(mat_dict.get("latent_heat", 0.0)) if base_phase == 1 else 0.0
    _ow(_ti_temp, AMBIENT_TEMP)
    _ow(_ti_heat_U, base_U)
    _ow(_ti_phase, base_phase)
    _ow(_ti_kappa_p, base_kappa)
    # mass = ρ · V_p  -  read per-particle V0 from the GPU field (correct over
    # the full N_PARTICLES range, unlike the entity-0-only V0 global).
    density = float(mat_dict.get("density", 1000.0))
    v0_all = _ti_v0.to_numpy()
    cur_m  = _ti_m.to_numpy()
    cur_m[abs_mask] = (density * v0_all[abs_mask]).astype(np.float32)
    _ti_m.from_numpy(cur_m)


def run_ficus_drop_compare_headless():
    """Side-by-side trained-PLY drop:
    left upright -> metal pot/base contacts first;
    right inverted -> jelly canopy contacts first.

    Both bodies use the same per-particle material identity: pot/base Metal,
    tree/canopy Jelly.  The second copy is only geometrically inverted.
    """
    import cv2 as _cv2
    import torch as _torch
    from gs_render import GSRenderer

    if not (FICUS_DROP_COMPARE or FICUS_METAL_COMPARE):
        raise ValueError("ficus comparison mode was not enabled")
    if SHAPE != "ply":
        raise ValueError("--ficus comparison requires --ply <trained 3DGS .ply>")
    if N_ENTITIES != 2:
        raise ValueError("ficus comparison requires exactly two entities")

    sims = []
    for ent_id, spawn in enumerate(SPAWN_POSITIONS):
        start = ent_id * N_PER_ENTITY
        end = start + N_PER_ENTITY
        s = GaussianBallSim(
            material_key="3",  # default tree/canopy = Jelly
            entity_id=ent_id,
            start=start, end=end,
            solver_name="ul_mlsmpm",
            spawn_position=np.array(spawn, dtype=np.float64),
            grid_h=_PER_ENTITY_GRIDH[ent_id],
        )
        s.drop()
        sims.append(s)

    # Start closer to ground than the default spawn, but leave enough fall time
    # that the viewer sees which side hits first.  Add a mild downward velocity
    # so impact happens inside the first third of the clip.
    cur_pos = _ti_pos.to_numpy()
    cur_vel = _ti_vel.to_numpy()
    for ent_id, s in enumerate(sims):
        sl = slice(s.start, s.end)
        min_y = float(cur_pos[sl, 1].min())
        cur_pos[sl, 1] += 1.55 - min_y
        cur_vel[sl, 1] = IMPACT_VY
    _ti_pos.from_numpy(cur_pos)
    _ti_vel.from_numpy(cur_vel)

    # Material paint.
    # Drop compare: left upright low-y metal pot/base, right inverted high-y metal
    # pot/base because the geometry was flipped.
    # Metal compare: left is bottom-metal/top-jelly, right is uniform metal.
    for ent_id, s in enumerate(sims):
        rest_y = REST_POS[s.start:s.end, 1]
        y_lo, y_hi = float(rest_y.min()), float(rest_y.max())
        frac = 0.50
        if FICUS_METAL_COMPARE and ent_id == 1:
            local_metal = np.ones(s.end - s.start, dtype=bool)
            label = "upright uniform Metal"
        elif ent_id == 0:
            local_metal = rest_y < (y_lo + frac * (y_hi - y_lo))
            label = "upright: low-y metal pot/base, jelly canopy above"
        else:
            local_metal = rest_y > (y_hi - frac * (y_hi - y_lo))
            label = "inverted: high-y metal pot/base, jelly canopy below"
        abs_mask = np.zeros(N_PARTICLES, dtype=bool)
        abs_mask[s.start:s.end] = local_metal
        _apply_material_to_mask(MATERIALS["2"], abs_mask)
        print(f"[ficus-compare] entity {ent_id}: {label}; "
              f"{int(local_metal.sum())}/{s.end-s.start} particles -> Metal, rest Jelly")

    # Preserve trained appearance for both copies.
    _opac = np.concatenate(_PLY_OPACITY_PER_ENTITY, axis=0).astype(np.float32)
    _shs = np.concatenate(_PLY_SHS_PER_ENTITY, axis=0).astype(np.float32)
    gs = GSRenderer(_opac, _shs, bg=(1.0, 1.0, 1.0))

    writer = None
    if VIDEO_OUT:
        fourcc = _cv2.VideoWriter_fourcc(*"mp4v")
        writer = _cv2.VideoWriter(VIDEO_OUT, fourcc, 30.0, (1280, 720))
        print(f"[ficus-compare] recording to {VIDEO_OUT}")

    eye = np.array([0.0, 3.0, 7.4], dtype=np.float32)
    target = np.array([0.0, 1.7, 0.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    dt_sub = DT / SUBSTEPS
    t_start = time.perf_counter()

    def _model_int(mat_model_str):
        if mat_model_str == "sand": return 2
        if mat_model_str == "fluid": return 1
        if mat_model_str == "stvk":  return 3
        return 0

    for frame in range(FRAMES):
        entity_descs = [
            {"start": s.start, "end": s.end,
             "kappa": float(s.mat["kappa"]), "mu_scale": s.mu_scale,
             "model": _model_int(s.mat["model"]),
             "floor_damp": s.floor_damp, "friction": s.friction}
            for s in sims
        ]
        for _ in range(SUBSTEPS):
            solver_ul.substep_multi(entity_descs, dt_sub)
        for s in sims:
            s._finish_frame()

        pos_t = _torch.cat([s.positions_torch() for s in sims], dim=0)
        cov_t = _torch.cat([s.covs_torch() for s in sims], dim=0)
        img = gs.render(positions=pos_t, covs=cov_t,
                        eye=eye, target=target, up=up,
                        fov_y_deg=42.0, width=1280, height=720)
        bgr = _cv2.cvtColor(img, _cv2.COLOR_RGB2BGR)
        if writer is not None:
            writer.write(bgr)
        else:
            _cv2.imshow("3DGS ficus drop compare", bgr)
            if _cv2.waitKey(1) == ord("q"):
                break

        if frame % 50 == 0:
            elapsed = time.perf_counter() - t_start
            y_mins = [float(s.positions[:, 1].min()) for s in sims]
            print(f"[ficus-compare] frame {frame}/{FRAMES} "
                  f"{(frame+1)/max(elapsed,1e-6):.1f} fps "
                  f"y_mins={[round(y, 2) for y in y_mins]}")

    if writer is not None:
        writer.release()
        print(f"[ficus-compare] done. video: {VIDEO_OUT}")
    else:
        _cv2.destroyAllWindows()


def run_collide_headless():
    """Multi-material collision demo  -  entities from _COLLIDE_ENTITIES on a
    shared UL world grid, gs_render output.

    Each entity is one body.  An entity may ALSO be split into two materials
    via its 4th spec field (split_material): the top half (rest-y > mid) gets
    overwritten with that material, demonstrating per-particle material
    identity WITHIN a single connected object.  Records to MP4 if --video set.
    """
    import cv2 as _cv2
    import torch as _torch
    from gs_render import GSRenderer
    if SCENE != "collide":
        raise ValueError("run_collide_headless requires --scene=collide")
    if not GS_RENDER_ENABLED:
        raise ValueError("run_collide_headless requires --gs-render or --video")

    # ---- Build entities ------------------------------------------------
    sims = []
    split_masks = {}   # ent_id -> boolean mask over entity slice (top half)
    for ent_id, spec in enumerate(_COLLIDE_ENTITIES):
        mat_key, shape, spawn = spec[0], spec[1], spec[2]
        split_mat = spec[3] if len(spec) > 3 else None
        start = ent_id * N_PER_ENTITY
        end   = start + N_PER_ENTITY
        s = GaussianBallSim(
            material_key=mat_key,
            entity_id=ent_id,
            start=start, end=end,
            solver_name="ul_mlsmpm",
            spawn_position=np.array(spawn, dtype=np.float64),
            grid_h=_PER_ENTITY_GRIDH[ent_id],
        )
        s.drop()
        sims.append(s)
        print(f"[collide] entity {ent_id}: {MATERIALS[mat_key]['name']:8s} "
              f"({shape:7s}) spawn={spawn}  N={N_PER_ENTITY}"
              + (f"  + top half = {MATERIALS[split_mat]['name']}" if split_mat else ""))

        # Split material: overwrite top half of this entity with a 2nd material.
        # This is per-particle material identity WITHIN one connected body.
        if split_mat and split_mat in MATERIALS:
            rest_y = REST_POS[start:end, 1]
            y_mid  = float(rest_y.mean())
            local_top = rest_y > y_mid           # mask over entity slice
            abs_mask  = np.zeros(N_PARTICLES, dtype=bool)
            abs_mask[start:end] = local_top
            _apply_material_to_mask(MATERIALS[split_mat], abs_mask)
            split_masks[ent_id] = local_top
            print(f"[collide]   split: {int(local_top.sum())}/{N_PER_ENTITY} "
                  f"top-half particles -> {MATERIALS[split_mat]['name']} "
                  f"(model={MATERIALS[split_mat]['model']})")

    # ---- Per-entity appearance for gs_render ---------------------------
    # Each entity's particle slice gets its own color (from mesh_color),
    # opacity (surface shell), and SH coefficients.
    _sh_c0 = 0.28209479
    _sh_c1 = 0.48860251
    _opac = np.zeros((N_PARTICLES, 1), dtype=np.float32)
    _shs  = np.zeros((N_PARTICLES, 16, 3), dtype=np.float32)

    def _hex_to_rgb(hexstr):
        h = hexstr.lstrip("#")
        return np.array([int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)])

    for ent_id, s in enumerate(sims):
        rest_slice = REST_POS[s.start:s.end]
        center = rest_slice.mean(axis=0)
        dist = np.linalg.norm(rest_slice - center, axis=1)
        max_r = float(dist.max()) + 1e-6
        depth = dist / max_r
        opac_raw = np.clip((depth - 0.80) / 0.20, 0.0, 1.0)
        _opac[s.start:s.end, 0] = (opac_raw * 0.95).astype(np.float32)

        # Per-particle base color.  If this entity is split, top half gets the
        # split material's color, bottom half gets the primary color.
        spec = _COLLIDE_ENTITIES[ent_id]
        split_mat = spec[3] if len(spec) > 3 else None
        rgb_bottom = _hex_to_rgb(s.mat.get("mesh_color", "#888888"))
        per_particle_rgb = np.tile(rgb_bottom, (len(rest_slice), 1))
        if split_mat and ent_id in split_masks:
            rgb_top = _hex_to_rgb(MATERIALS[split_mat].get("mesh_color", "#888888"))
            per_particle_rgb[split_masks[ent_id]] = rgb_top

        # SH DC from per-particle color, band 1 = directional lighting
        _shs[s.start:s.end, 0, :] = ((per_particle_rgb - 0.5) / _sh_c0).astype(np.float32)
        _shs[s.start:s.end, 2, :] = (0.5 / _sh_c1)
        _shs[s.start:s.end, 1, :] = (0.10 / _sh_c1)
        _shs[s.start:s.end, 3, :] = (0.05 / _sh_c1)

        n_shell = int((opac_raw > 0.01).sum())
        if split_mat:
            print(f"[collide]   appearance: bottom={rgb_bottom.round(2)} "
                  f"top={_hex_to_rgb(MATERIALS[split_mat]['mesh_color']).round(2)} "
                  f"shell={n_shell}/{N_PER_ENTITY}")
        else:
            print(f"[collide]   appearance: color={rgb_bottom.round(2)} "
                  f"shell={n_shell}/{N_PER_ENTITY}")

    # Override rest covariances to surface-aligned flat discs  -  better look.
    # Physics never reads _ti_rcov so this is appearance-only.
    cur_rcov = _ti_rcov.to_numpy()
    for s in sims:
        rest_slice = REST_POS[s.start:s.end]
        cur_rcov[s.start:s.end] = make_base_covs(rest_slice,
                                                  radial_spread=0.02,
                                                  tangent_spread=0.18).astype(np.float32)
    _ti_rcov.from_numpy(cur_rcov)
    _ti_update_covs()

    gs = GSRenderer(_opac, _shs, bg=(1.0, 1.0, 1.0))

    # ---- Video writer or display ---------------------------------------
    if VIDEO_OUT:
        fourcc = _cv2.VideoWriter_fourcc(*"mp4v")
        writer = _cv2.VideoWriter(VIDEO_OUT, fourcc, 30.0, (1280, 720))
        print(f"[collide] recording to {VIDEO_OUT} @ 30fps (1280×720)")
    else:
        writer = None
        _cv2.namedWindow("3DGS multi-material collide", _cv2.WINDOW_NORMAL)
        _cv2.resizeWindow("3DGS multi-material collide", 1280, 720)

    # Camera  -  hero shot framing
    eye    = np.array([3.0,  4.0, 7.0], dtype=np.float32)
    target = np.array([0.0,  2.5, 0.0], dtype=np.float32)
    up     = np.array([0.0,  1.0, 0.0], dtype=np.float32)

    # ---- Main loop -----------------------------------------------------
    dt_sub = DT / SUBSTEPS
    t_start = time.perf_counter()

    # Pre-compute per-entity material descriptors for substep_multi
    def _model_int(mat_model_str):
        if mat_model_str == "sand": return 2
        if mat_model_str == "fluid": return 1
        if mat_model_str == "stvk":  return 3
        return 0

    for frame in range(FRAMES):
        # Build entity descriptors (kappa/floor_damp/friction may change if
        # apply_material is called mid-run; rebuild each frame to be safe).
        entity_descs = [
            {"start": s.start, "end": s.end,
             "kappa": float(s.mat["kappa"]), "mu_scale": s.mu_scale,
             "model": _model_int(s.mat["model"]),
             "floor_damp": s.floor_damp, "friction": s.friction}
            for s in sims
        ]
        # 40 shared-grid substeps
        for _ in range(SUBSTEPS):
            solver_ul.substep_multi(entity_descs, dt_sub)
        # Per-entity post-processing (polar decomp, covs update, sync to CPU)
        for s in sims:
            s._finish_frame()

        # Render
        try:
            pos_t = _torch.cat([s.positions_torch() for s in sims], dim=0)
            cov_t = _torch.cat([s.covs_torch()      for s in sims], dim=0)
            img = gs.render(positions=pos_t, covs=cov_t,
                            eye=eye, target=target, up=up,
                            fov_y_deg=42.0, width=1280, height=720)
            bgr = _cv2.cvtColor(img, _cv2.COLOR_RGB2BGR)
            if writer is not None:
                writer.write(bgr)
            else:
                _cv2.imshow("3DGS multi-material collide", bgr)
                if _cv2.waitKey(1) == ord("q"):
                    break
        except Exception as e:
            print(f"[collide] render frame {frame} failed: {e}")
            break

        if frame % 30 == 0:
            elapsed = time.perf_counter() - t_start
            fps = (frame + 1) / max(elapsed, 1e-6)
            y_mins = [float(s.positions[:, 1].min()) for s in sims]
            thermo_bits = []
            if THERMO_ENABLED:
                for si, s in enumerate(sims):
                    if "melt_temp" in s.mat:
                        thermo_bits.append(
                            f"s{si}:molten={s.thermo_molten}/{s.end-s.start},"
                            f"Tmax={s.thermo_temp_max:.1f}")
            thermo_msg = f"  thermo={thermo_bits}" if thermo_bits else ""
            print(f"[collide] frame {frame}/{FRAMES}  {fps:.1f} fps  "
                  f"y_mins={[round(y, 2) for y in y_mins]}{thermo_msg}")

    if writer is not None:
        writer.release()
        print(f"[collide] done. video: {VIDEO_OUT}")
    else:
        _cv2.destroyAllWindows()


def run_physgaussian_ficus_match():
    """Run the stock PhysGaussian ficus setup and write solver-state checkpoints."""
    if not PLY_PATH:
        raise ValueError("--physgaussian-ficus-match requires --ply")
    if SOLVER != "ul_mlsmpm":
        raise ValueError("--physgaussian-ficus-match requires --solver ul_mlsmpm")
    if PHYSGAUSSIAN_FICUS_DROP and not VIDEO_OUT:
        raise ValueError("--physgaussian-ficus-drop requires --video")
    if not PHYSGAUSSIAN_FICUS_DROP and not STATE_OUTPUT:
        raise ValueError("--physgaussian-ficus-match requires --state-output")

    out_dir = Path(STATE_OUTPUT) if STATE_OUTPUT else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    sim = GaussianBallSim(material_key="1")
    sim.drop()

    # PhysGaussian's additional_material_params override density in the upper
    # z half of the transformed scene. E and nu are unchanged in that region.
    pos = _ti_pos.to_numpy()[sim.start:sim.end]
    upper = np.all(
        np.abs(pos - np.array([1.0, 1.0, 1.5], np.float32))
        < np.array([1.0, 1.0, 0.5], np.float32),
        axis=1,
    )
    density = (
        np.full(len(pos), 200.0, np.float32)
        if PHYSGAUSSIAN_FICUS_DROP
        else np.where(upper, 70.0, 200.0).astype(np.float32)
    )
    mass = density * _ti_v0.to_numpy()[sim.start:sim.end]
    full_mass = _ti_m.to_numpy()
    full_mass[sim.start:sim.end] = mass
    _ti_m.from_numpy(full_mass)

    sample_frames = {0, 1, 2, 5, 10, 25, 50, 75, 100, FRAMES}
    sample_frames = {f for f in sample_frames if 0 <= f <= FRAMES}

    def save_state(frame):
        if out_dir is None:
            return
        sl = slice(sim.start, sim.end)
        np.savez_compressed(
            out_dir / f"state_{frame:04d}.npz",
            x=_ti_pos.to_numpy()[sl],
            v=_ti_vel.to_numpy()[sl],
            F=_ti_F.to_numpy()[sl],
            C=_ti_C.to_numpy()[sl],
        )
        print(f"[phys-match] saved frame {frame}/{FRAMES}")

    renderer = None
    if PHYSGAUSSIAN_FICUS_DROP:
        from matched_ficus_video import MatchedFicusRenderer
        renderer = MatchedFicusRenderer(PLY_PATH, VIDEO_OUT, fps=30.0, size=800)
        renderer.write(pos, "Representation UL-MLS-MPM", 0)
    else:
        save_state(0)

    # The stock impulse acts for exactly the first substep, before P2G.
    if not PHYSGAUSSIAN_FICUS_DROP:
        impulse_mask = np.all(
            np.abs(pos - np.array([1.0, 1.0, 1.0], np.float32)) < 1.0, axis=1
        )
        vel = _ti_vel.to_numpy()
        vel_slice = vel[sim.start:sim.end]
        vel_slice[impulse_mask, 0] += (-0.18 / mass[impulse_mask]) * 1.0e-4
        vel[sim.start:sim.end] = vel_slice
        _ti_vel.from_numpy(vel)

    dt_sub = 1.0e-4
    for frame in range(1, FRAMES + 1):
        for _ in range(SUBSTEPS):
            solver_ul.substep(
                sim.start, sim.end, dt_sub,
                kappa=float(sim.mat["kappa"]),
                mu_scale=1.0,
                model=0,
                floor_damp=0.0,
                friction=0.0,
                use_floor=False,
                physgaussian_ficus_bc=True,
            )
        if renderer is not None:
            renderer.write(
                _ti_pos.to_numpy()[sim.start:sim.end],
                "Representation UL-MLS-MPM",
                frame,
            )
            if frame % 10 == 0:
                print(f"[ficus-drop] rendered frame {frame}/{FRAMES}")
        elif frame in sample_frames:
            save_state(frame)

    if renderer is not None:
        renderer.close()
        print(f"[ficus-drop] video: {VIDEO_OUT}")
        return

    metadata = {
        "backend": "representation_ul_mlsmpm",
        "particles": int(sim.end - sim.start),
        "frames": int(FRAMES),
        "frame_dt": 0.04,
        "substep_dt": 1.0e-4,
        "substeps_per_frame": 400,
        "density_70_count": int(upper.sum()),
        "density_200_count": int((~upper).sum()),
        "total_volume": float(_ti_v0.to_numpy()[sim.start:sim.end].sum()),
        "total_mass": float(mass.sum()),
        "sample_frames": sorted(sample_frames),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"[phys-match] complete: {out_dir}")


if __name__ == "__main__":
    if PHYSGAUSSIAN_FICUS_MATCH:
        run_physgaussian_ficus_match()
    elif (FICUS_DROP_COMPARE or FICUS_METAL_COMPARE) and GS_RENDER_ENABLED:
        run_ficus_drop_compare_headless()
    elif SCENE == "collide" and GS_RENDER_ENABLED:
        run_collide_headless()
    elif GS_ONLY or VIDEO_OUT:
        run_gs_headless()
    elif SCENE == "compare":
        run_compare()
    else:
        run()
