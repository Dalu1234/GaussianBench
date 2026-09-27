"""
solver_ul.py
Updated-Lagrangian MLS-MPM solver  -  PhysGaussian-style baseline.

What makes this UL (vs the TL solver in solver_tl.py):
  1.  Grid is REBUILT EVERY SUBSTEP in the current (deformed) configuration.
      Each particle's 27-node stencil indices, B-spline weights w_ip, and
      world-frame gradients ∇_x w_ip are recomputed against the particle's
      current world position.  Cf. TL where these are precomputed once
      against the rest pose and never touched again.

  2.  F update is MULTIPLICATIVE:
        F_new = (I + Δt · ∇v_p) · F_old
      where ∇v_p = Σ_i v_i ⊗ ∇_x w_i(x_p) (gathered with world-frame ∇w).
      Cf. TL where F += dt · C (additive).

  3.  Stress is scattered as KIRCHHOFF τ with world-frame gradients:
        f_i = -V₀ · τ · ∇_x w_i(x_p)
      where τ = P · F^T and V₀ is the rest volume per particle.
      (This is mathematically the standard UL P2G force: V·σ·∇N = V₀·τ·∇N
       since V = J·V₀ and σ = τ/J.)
      Cf. TL where we scatter PK1 P with rest-frame gradients:
        f_i = -V₀ · P · ∇_X w_i(X_p)

  4.  No per-frame WLS F recompute  -  that's a TL drift correction; UL just
      accumulates the multiplicative update.

Everything ELSE is identical to TL by design  -  same particle struct, same
constitutive law (corotated PK1, then post-multiply by F^T to get τ), same
SVD clamp bounds, same MPM_STRESS_CLAMP, same particle-level floor contact,
same APIC C transfer.  That's the whole point of the comparison.
"""
import numpy as np
import taichi as ti

import gaussian_mesh_poc as G   # particle fields + constants + constitutive @ti.funcs


# ============================================================
# UL grid fields  -  world-frame, rebuilt every substep
# ============================================================

# World-grid domain.  Must be wide enough to enclose any plausible particle
# motion across the run.  Origin and extent are fixed at module load (we
# Do not resize per substep. Reallocation would be prohibitively expensive on
# the GPU and is unnecessary because the upper bound is fixed at initialization.
# what PhysGaussian does either; they just allocate a big enough domain).
if G.GRID_LIM_OVERRIDE is not None:
    _grid_lim = float(G.GRID_LIM_OVERRIDE)
    if G.PHYSGAUSSIAN_FICUS_MATCH:
        _UL_GRID_ORIGIN = np.zeros(3, dtype=np.float32)
    else:
        _UL_GRID_ORIGIN = np.array([-0.5 * _grid_lim, 0.0, -0.5 * _grid_lim], dtype=np.float32)
    _UL_GRID_EXTENT = np.array([_grid_lim, _grid_lim, _grid_lim], dtype=np.float32)
else:
    _UL_GRID_ORIGIN = np.array([-6.0, -1.0, -6.0], dtype=np.float32)
    _UL_GRID_EXTENT = np.array([12.0, 12.0, 12.0], dtype=np.float32)
_UL_GRID_DX     = float(G.MPM_GRID_H)                              # match TL spacing for fair comparison
_UL_GRID_N      = (int(round(max(_UL_GRID_EXTENT) / _UL_GRID_DX))
                   if G.PHYSGAUSSIAN_FICUS_MATCH
                   else int(np.ceil(max(_UL_GRID_EXTENT) / _UL_GRID_DX)) + 1)
_UL_TOTAL_CELLS = _UL_GRID_N ** 3

# Memory ceiling for the UL world grid.  At 4 fields × (1 float + 3 vec3) = 40
# bytes/cell each, a 150³ grid is ~3.4M cells = ~540 MB GPU.  Plenty for any
# reasonable scene on consumer hardware.  Bigger than that and we should
# probably switch to a sparse representation rather than blindly allocate.
_HARD_GRID_CAP = 150 ** 3
assert _UL_TOTAL_CELLS <= _HARD_GRID_CAP, (
    f"UL grid would need {_UL_TOTAL_CELLS} nodes "
    f"(more than {_HARD_GRID_CAP} cap).  Either coarsen MPM_GRID_H or "
    f"shrink _UL_GRID_EXTENT in solver_ul.py."
)

_MAX_GRID_UL = max(_UL_TOTAL_CELLS, 32768)

# Runtime-movable grid origin (co-moving domain support).  The stencil
# builder reads THIS field instead of the compile-time constant, so a run
# harness can translate the domain to follow a drifting body between
# substeps.  This is exact, not an approximation: the world grid carries no
# state across substeps (cleared and rebuilt every substep), so only the
# node positions move.  Defaults to the static origin  -  behavior is
# unchanged unless set_grid_origin() is called.  Only valid for scenes with
# no floor anchor, no floor BC, and no thermal floor source: those kernels
# stay anchored to the STATIC origin on purpose, since a floor that follows
# the body would be meaningless.
_ti_grid_origin = ti.Vector.field(3, dtype=ti.f32, shape=())
_ti_grid_origin[None] = [float(_UL_GRID_ORIGIN[0]),
                         float(_UL_GRID_ORIGIN[1]),
                         float(_UL_GRID_ORIGIN[2])]


def set_grid_origin(origin) -> None:
    """Move the world-grid domain so it spans origin + [0, extent]^3.

    For co-moving runs: call between substeps (or once per display frame)
    with, for example, body_center - 0.5 * extent per axis.  Do NOT use
    together with the floor (init_floor / use_floor=True) or the thermal
    hot floor; those are anchored to the static module-level origin.
    """
    _ti_grid_origin[None] = [float(origin[0]), float(origin[1]),
                             float(origin[2])]

# Grid scratch (cleared per substep, then accumulated, then read).
# Precision note (2026-07, benchmark investigation): these were trialed at
# f64 to chase a linear-in-time momentum drift on the GaussianBench A1
# scenes. Measured effect: none. The drift scales with substep count and
# sits exactly at the f32 epsilon of the stored PARTICLE velocities, so the
# floor is the f32 particle state, not the grid accumulation; f64 here buys
# nothing and f64 atomics are slow on consumer GPUs. Kept at f32.
_ti_gm_ul  = ti.field(ti.f32, shape=_MAX_GRID_UL)
_ti_gmo_ul = ti.Vector.field(3, dtype=ti.f32, shape=_MAX_GRID_UL)
_ti_gf_ul  = ti.Vector.field(3, dtype=ti.f32, shape=_MAX_GRID_UL)
_ti_gv_ul  = ti.Vector.field(3, dtype=ti.f32, shape=_MAX_GRID_UL)

# Persistent anchor mass  -  the floor is a layer of static heavy particles
# that scatter their mass ONCE into this field at startup.  Never cleared.
# Read in grid_update alongside body mass: m_total = m_body + m_floor.
_ti_floor_gm_ul = ti.field(ti.f32, shape=_MAX_GRID_UL)

# Temporary buffers used during the one-shot floor scatter at init.
_FLOOR_MAX_PARTICLES = 65536
_ti_floor_pos = ti.Vector.field(3, dtype=ti.f32, shape=_FLOOR_MAX_PARTICLES)
_ti_floor_pm  = ti.field(ti.f32, shape=_FLOOR_MAX_PARTICLES)

# Per-particle 27-node stencil, rewritten every substep
_ti_nidx_ul = ti.field(ti.i32,            shape=(G.N_PARTICLES, 27))
_ti_wip_ul  = ti.field(ti.f32,            shape=(G.N_PARTICLES, 27))
_ti_dwip_ul = ti.Vector.field(3, ti.f32,  shape=(G.N_PARTICLES, 27))  # world-frame ∇_x w
_ti_dpos_ul = ti.Vector.field(3, ti.f32,  shape=(G.N_PARTICLES, 27))  # node_world_pos − particle_pos

# ---- Thermodynamics: grid heat fields (Stomakhin 2014, per-frame heat solve) ----
_ti_gTmass_ul = ti.field(ti.f32, shape=_MAX_GRID_UL)   # Σ w·m   (heat rasterization mass)
_ti_gTm_ul    = ti.field(ti.f32, shape=_MAX_GRID_UL)   # Σ w·m·T (temperature·mass accumulator)
_ti_gT_ul     = ti.field(ti.f32, shape=_MAX_GRID_UL)   # grid temperature T_i = gTm/gTmass
_ti_gT_new_ul = ti.field(ti.f32, shape=_MAX_GRID_UL)   # diffusion double-buffer

# ---- Chorin pressure projection fields (per-substep, fluid cells only) ----
_ti_gfluid_m_ul = ti.field(ti.f32, shape=_MAX_GRID_UL) # mass from fluid (phase==1) particles
_ti_gp_ul       = ti.field(ti.f32, shape=_MAX_GRID_UL) # pressure p (Poisson solution)
_ti_gp_new_ul   = ti.field(ti.f32, shape=_MAX_GRID_UL) # pressure double-buffer for G-S
_ti_gdiv_ul     = ti.field(ti.f32, shape=_MAX_GRID_UL) # ∇·v of grid velocity field


# ============================================================
# Helpers
# ============================================================

@ti.func
def _bspline2_w_dw(s):
    """Quadratic B-spline weight and its derivative for offset s.
    s is the fractional offset (particle position − node position) / dx.
    Returns (w, dw_ds). Multiply dw_ds by 1/dx to get the world-frame ∇w_x."""
    a = ti.abs(s)
    w  = 0.0
    dw = 0.0
    if a < 0.5:
        w  =  0.75 - a * a
        dw = -2.0 * s
    elif a < 1.5:
        # piecewise quadratic on |s|∈[0.5, 1.5]
        t  = 1.5 - a
        w  = 0.5 * t * t
        # d/ds of 0.5 (1.5 − |s|)² = -(1.5 − |s|) · sign(s)
        dw = -(1.5 - a) * (1.0 if s > 0.0 else -1.0)
    # else: w = 0, dw = 0 (outside support)
    return w, dw


@ti.func
def _ul_cell_flat(ix, iy, iz):
    """Flat index into the world UL grid. Matches the (nx, ny, nz) ordering
    used by the build kernel; both must agree for indices to round-trip."""
    return ix * _UL_GRID_N * _UL_GRID_N + iy * _UL_GRID_N + iz


@ti.func
def _positive_det_F_increment(M):
    """Truncated exponential update from Stomakhin 2014 Sec. 5.9."""
    eye = ti.Matrix.identity(ti.f32, 3)
    A = eye + M
    if A.determinant() <= 0.0:
        B = eye + 0.5 * M
        if B.determinant() > 0.0:
            A = B @ B
        else:
            C = eye + 0.25 * M
            if C.determinant() > 0.0:
                A = C @ C @ C @ C
            else:
                D = eye + 0.125 * M
                A = D @ D @ D @ D @ D @ D @ D @ D
    return A


# ============================================================
# Kernels
# ============================================================

@ti.kernel
def _ti_clear_grid_ul():
    """Zero ALL UL grid nodes  -  cheap since they're a flat array."""
    for i in range(_MAX_GRID_UL):
        _ti_gm_ul[i]  = 0.0
        _ti_gmo_ul[i] = ti.Vector([0.0, 0.0, 0.0])
        _ti_gf_ul[i]  = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def _ti_build_world_grid_ul(start: ti.i32, end: ti.i32):
    """For each particle in [start, end): compute base grid node + 27-node
    quadratic B-spline stencil (indices, weights, world-frame gradients,
    displacements) AT THE CURRENT WORLD POSITION.  Writes _ti_nidx_ul,
    _ti_wip_ul, _ti_dwip_ul, _ti_dpos_ul for this entity.

    Quadratic B-spline base = floor(x/dx − 0.5).  Stencil offsets {0,1,2}."""
    inv_dx = 1.0 / _UL_GRID_DX
    origin = _ti_grid_origin[None]
    for p in range(start, end):
        xp = G._ti_pos[p]
        # particle position in fractional grid units relative to origin
        xg = (xp - origin) * inv_dx
        # base = lower corner of 3x3x3 stencil
        bx = ti.cast(ti.floor(xg[0] - 0.5), ti.i32)
        by = ti.cast(ti.floor(xg[1] - 0.5), ti.i32)
        bz = ti.cast(ti.floor(xg[2] - 0.5), ti.i32)
        k = 0
        for ox in ti.static(range(3)):
            for oy in ti.static(range(3)):
                for oz in ti.static(range(3)):
                    ix = bx + ox
                    iy = by + oy
                    iz = bz + oz
                    # fractional offset from particle to this node, in grid units
                    sx = xg[0] - ti.cast(ix, ti.f32)
                    sy = xg[1] - ti.cast(iy, ti.f32)
                    sz = xg[2] - ti.cast(iz, ti.f32)
                    wx, dwx = _bspline2_w_dw(sx)
                    wy, dwy = _bspline2_w_dw(sy)
                    wz, dwz = _bspline2_w_dw(sz)
                    w = wx * wy * wz
                    # world-frame ∇w  =  (dw/ds) · (1/dx) along each axis
                    dw_x = (dwx * wy * wz) * inv_dx
                    dw_y = (wx * dwy * wz) * inv_dx
                    dw_z = (wx * wy * dwz) * inv_dx
                    # node world position
                    nwp = origin + ti.Vector([ti.cast(ix, ti.f32),
                                              ti.cast(iy, ti.f32),
                                              ti.cast(iz, ti.f32)]) * _UL_GRID_DX
                    # boundary guard  -  zero out weight if outside domain
                    in_bounds = (
                        ix >= 0 and ix < _UL_GRID_N and
                        iy >= 0 and iy < _UL_GRID_N and
                        iz >= 0 and iz < _UL_GRID_N
                    )
                    if not in_bounds:
                        w = 0.0
                        dw_x = 0.0; dw_y = 0.0; dw_z = 0.0
                        # store a valid index so atomic_add doesn't oob; weight is zero
                        ix = 0; iy = 0; iz = 0
                    _ti_nidx_ul[p, k] = _ul_cell_flat(ix, iy, iz)
                    _ti_wip_ul[p, k]  = w
                    _ti_dwip_ul[p, k] = ti.Vector([dw_x, dw_y, dw_z])
                    _ti_dpos_ul[p, k] = nwp - xp
                    k += 1
        # Enforce the partition-of-unity identities structurally: in exact
        # arithmetic sum_k w = 1 and sum_k grad_w = 0, but f32 rounding
        # leaves consistent-sign residuals (~1e-7). Because a particle's
        # stencil geometry barely changes between substeps, the residual is
        # a FROZEN bias, and through p2g it becomes a constant spurious
        # net force / mass leak per particle: measured on the benchmark A1
        # scenes as a linear-in-time momentum drift. Renormalizing w and
        # projecting out the grad_w sum enforces discrete Newton's third
        # law per particle at any float precision. Skipped when the
        # boundary guard clipped the stencil (wsum well below 1), where the
        # deficit is intentional.
        wsum = 0.0
        dwsum = ti.Vector([0.0, 0.0, 0.0])
        for kk in ti.static(range(27)):
            wsum += _ti_wip_ul[p, kk]
            dwsum += _ti_dwip_ul[p, kk]
        if wsum > 0.99:
            inv_wsum = 1.0 / wsum
            for kk in ti.static(range(27)):
                w_n = _ti_wip_ul[p, kk] * inv_wsum
                _ti_wip_ul[p, kk] = w_n
                _ti_dwip_ul[p, kk] = _ti_dwip_ul[p, kk] - dwsum * w_n
            # First-moment (linear) consistency: sum_k w dpos = 0 exactly
            # for B-splines, but the f32 residual eps couples with the APIC
            # affine matrix in p2g (momentum term m C @ dpos), injecting a
            # constant force m C @ eps per particle. For a spinning body
            # C = skew(omega), so the bias sits in the rotation plane with
            # a frozen direction, matching the measured linear momentum
            # drift. Subtracting the weighted mean from every dpos restores
            # the identity at machine precision (valid since sum w = 1).
            dpsum = ti.Vector([0.0, 0.0, 0.0])
            for kk in ti.static(range(27)):
                dpsum += _ti_wip_ul[p, kk] * _ti_dpos_ul[p, kk]
            for kk in ti.static(range(27)):
                _ti_dpos_ul[p, kk] = _ti_dpos_ul[p, kk] - dpsum


@ti.kernel
def _ti_p2g_ul(start: ti.i32, end: ti.i32,
               kappa: ti.f32, mu_scale: ti.f32, model: ti.i32):
    """Particle-to-grid scatter using world-frame gradients and Kirchhoff τ.
    Force: f_i = -V₀ · τ · ∇_x w_i(x_p) where τ = P · F^T.
    APIC momentum: m_p · (v_p + C_p · dp) scattered with weight w."""
    for p in range(start, end):
        Fe  = G._ti_F[p] @ G._ti_Fp[p].inverse()   # elastic F (same as TL)
        v   = G._ti_vel[p]
        C   = G._ti_C[p]
        m   = G._ti_m[p]
        # Damage-softened mu/lam  -  fully damaged (d=1) contributes zero stress
        mu  = G._mu_eff_sr(p, mu_scale) * (1.0 - G._ti_damage[p])
        lam = G._ti_lam[p] * (1.0 - G._ti_damage[p])
        P   = ti.Matrix.zero(ti.f32, 3, 3)
        # Per-particle constitutive law dispatch  -  reads _ti_model[p] so
        # different particles in the same body can use different stress
        # equations (corotated vs Drucker-Prager vs fluid).
        p_model = G._ti_model[p]
        if p_model == 0:
            P = G._pk1_corotated(Fe, mu, lam)
        elif p_model == 2:
            P = G._pk1_sand(Fe, mu, lam)
        elif p_model == 3:
            # StVK metal: stress computed from F directly (not Fe), because
            # the log-strain return map writes F_elastic into F (Fp=I).
            P = G._pk1_stvk(G._ti_F[p], mu, lam)
        else:
            kp = kappa
            if G._ti_kappa_p[p] > 0.0:
                kp = G._ti_kappa_p[p]
            P = G._pk1_fluid(Fe, kp, mu)
        # Cap stress norm  -  same numerical safety net as TL, applied to PK1
        # BEFORE conversion to Kirchhoff so the bound has the same physical
        # meaning in both solvers.
        pn = P.norm()
        if pn > G.MPM_STRESS_CLAMP:
            P = P * (G.MPM_STRESS_CLAMP / pn)
        # τ = P · F^T   -  Kirchhoff stress for world-frame integration
        tau = P @ Fe.transpose()
        V0  = G._ti_v0[p]
        for k in ti.static(range(27)):
            idx = _ti_nidx_ul[p, k]
            w   = _ti_wip_ul[p, k]
            dw  = _ti_dwip_ul[p, k]
            dp  = _ti_dpos_ul[p, k]
            ti.atomic_add(_ti_gmo_ul[idx], w * m * (v + C @ dp))
            ti.atomic_add(_ti_gm_ul[idx],  w * m)
            ti.atomic_add(_ti_gf_ul[idx],  -V0 * tau @ dw)


@ti.kernel
def _ti_apply_floor_bc_ul(floor_y: ti.f32):
    """Grid-level separating BC at world y = floor_y (PhysGaussian-style).
    For every grid node within the contact band with downward velocity, zero
    the y-component (leave tangential alone).  This is a velocity *constraint*
     -  unconditionally stable regardless of K, dt, or N.

    Band extends 2·dx ABOVE the floor to cover the quadratic B-spline stencil
    width (a particle at the floor has stencil nodes at y ∈ {floor−dx, floor,
    floor+dx}; clamping only nodes strictly below would leak unclamped
    downward velocity through the upper stencil layers).

    Tangential drag is handled per-particle in g2p (Coulomb friction)."""
    origin_y = float(_UL_GRID_ORIGIN[1])
    dx       = _UL_GRID_DX
    band_top = floor_y + 2.0 * dx
    for i in range(_MAX_GRID_UL):
        iy = (i // _UL_GRID_N) % _UL_GRID_N
        node_y = origin_y + ti.cast(iy, ti.f32) * dx
        if node_y <= band_top:
            v = _ti_gv_ul[i]
            if v[1] < 0.0:
                _ti_gv_ul[i] = ti.Vector([v[0], 0.0, v[2]])


@ti.kernel
def _ti_apply_physgaussian_ficus_cuboid_bc():
    """Stock ficus_config.json fixed cuboid grid-velocity constraint."""
    origin = ti.Vector([
        float(_UL_GRID_ORIGIN[0]),
        float(_UL_GRID_ORIGIN[1]),
        float(_UL_GRID_ORIGIN[2]),
    ])
    center = ti.Vector([1.0, 1.0, 0.5])
    half_size = ti.Vector([0.5, 0.5, 0.28])
    for i in range(_MAX_GRID_UL):
        ix = i // (_UL_GRID_N * _UL_GRID_N)
        iy = (i // _UL_GRID_N) % _UL_GRID_N
        iz = i % _UL_GRID_N
        node = origin + _UL_GRID_DX * ti.Vector([ix, iy, iz])
        offset = node - center
        if (ti.abs(offset[0]) < half_size[0]
                and ti.abs(offset[1]) < half_size[1]
                and ti.abs(offset[2]) < half_size[2]):
            _ti_gv_ul[i] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def _ti_apply_physgaussian_bounding_box_bc():
    """PhysGaussian add_bounding_box(), including its three-node floor band."""
    padding = 3
    for i in range(_MAX_GRID_UL):
        ix = i // (_UL_GRID_N * _UL_GRID_N)
        iy = (i // _UL_GRID_N) % _UL_GRID_N
        iz = i % _UL_GRID_N
        v = _ti_gv_ul[i]
        if ix < padding and v[0] < 0.0:
            v[0] = 0.0
        if ix >= _UL_GRID_N - padding and v[0] > 0.0:
            v[0] = 0.0
        if iy < padding and v[1] < 0.0:
            v[1] = 0.0
        if iy >= _UL_GRID_N - padding and v[1] > 0.0:
            v[1] = 0.0
        if iz < padding and v[2] < 0.0:
            v[2] = 0.0
        if iz >= _UL_GRID_N - padding and v[2] > 0.0:
            v[2] = 0.0
        _ti_gv_ul[i] = v


@ti.kernel
def _ti_grid_update_ul(dt: ti.f32):
    """Newton on the world grid with floor as a second object.

    The grid sees TWO contributions to mass at each node:
      - m_body  = _ti_gm_ul[i]        (live body mass, cleared+rescattered each substep)
      - m_floor = _ti_floor_gm_ul[i]  (static anchor mass, precomputed at init)

    Standard mass-weighted velocity normalization:
        gv = (body_momentum + dt·(body_stress + m_body·g)) / (m_body + m_floor)

    Gravity is applied only to body mass  -  the floor is rigid, doesn't fall.
    When a node is shared between body and floor (impact contact), the floor's
    huge mass dilutes body velocity toward zero, exactly as two real objects
    sharing momentum would.  When m_floor=0, this reduces to standard MPM.

    No velocity projection, no BC kernel, no "force canceller".  The floor
    is just another body in the shared world grid  -  the same way two MPM
    bodies interact via their shared grid contributions.
    """
    grav = ti.Vector(
        [0.0, 0.0, -5.0]
        if G.PHYSGAUSSIAN_FICUS_DROP
        else [0.0, G.GRAVITY_Y, 0.0]
    )
    for i in range(_MAX_GRID_UL):
        m_body  = _ti_gm_ul[i]
        m_floor = _ti_floor_gm_ul[i]
        m_total = m_body + m_floor
        if m_total > 1e-12:
            _ti_gv_ul[i] = G.GRID_V_DAMPING_SCALE * (
                (_ti_gmo_ul[i] + dt * (_ti_gf_ul[i] + m_body * grav)) / m_total
            )
        else:
            _ti_gv_ul[i] = ti.Vector([0.0, 0.0, 0.0])


# ---- Floor (anchor body) scatter  -  runs ONCE at init ----

@ti.kernel
def _ti_clear_floor_gm_ul():
    for i in range(_MAX_GRID_UL):
        _ti_floor_gm_ul[i] = 0.0


@ti.kernel
def _ti_scatter_floor_to_grid(n_floor: ti.i32):
    """Scatter floor particle mass to the persistent anchor mass field using
    the same quadratic B-spline stencil the body uses.  No momentum (floor
    velocity = 0), no stress (floor is rigid).  Called once at init."""
    inv_dx = 1.0 / _UL_GRID_DX
    origin = ti.Vector([_UL_GRID_ORIGIN[0], _UL_GRID_ORIGIN[1], _UL_GRID_ORIGIN[2]])
    for p in range(n_floor):
        xp = _ti_floor_pos[p]
        m  = _ti_floor_pm[p]
        xg = (xp - origin) * inv_dx
        bx = ti.cast(ti.floor(xg[0] - 0.5), ti.i32)
        by = ti.cast(ti.floor(xg[1] - 0.5), ti.i32)
        bz = ti.cast(ti.floor(xg[2] - 0.5), ti.i32)
        for ox in ti.static(range(3)):
            for oy in ti.static(range(3)):
                for oz in ti.static(range(3)):
                    ix = bx + ox
                    iy = by + oy
                    iz = bz + oz
                    in_bounds = (
                        ix >= 0 and ix < _UL_GRID_N and
                        iy >= 0 and iy < _UL_GRID_N and
                        iz >= 0 and iz < _UL_GRID_N
                    )
                    if in_bounds:
                        sx = xg[0] - ti.cast(ix, ti.f32)
                        sy = xg[1] - ti.cast(iy, ti.f32)
                        sz = xg[2] - ti.cast(iz, ti.f32)
                        wx, _ = _bspline2_w_dw(sx)
                        wy, _ = _bspline2_w_dw(sy)
                        wz, _ = _bspline2_w_dw(sz)
                        w = wx * wy * wz
                        idx = _ul_cell_flat(ix, iy, iz)
                        ti.atomic_add(_ti_floor_gm_ul[idx], w * m)


def init_floor(floor_y: float, mass_per_particle: float = 1.0,
               n_per_axis: int = 120, extent: float = 10.0) -> None:
    """Initialise the floor as a 2D layer of static heavy particles spread
    across [-extent/2, +extent/2] in x and z, at world y = floor_y.

    n_per_axis × n_per_axis particles total.  Each particle contributes
    mass_per_particle to the grid via standard B-spline scatter.

    Mass sizing rule of thumb: pick mass_per_particle ≫ body_particle_mass
    so the grid average at contact is dominated by the floor.  With body
    m_p ~ 1e-5 kg and mass_per_particle=1 kg, the ratio is 1e5 → body
    velocity at floor nodes is reduced to ~v_body / 1e5 ≈ 0.

    Runs the scatter once and stores the result in the persistent
    _ti_floor_gm_ul field; substep cost from the floor is then zero."""
    n_floor = n_per_axis * n_per_axis
    assert n_floor <= _FLOOR_MAX_PARTICLES, (
        f"floor needs {n_floor} particles, exceeds _FLOOR_MAX_PARTICLES "
        f"({_FLOOR_MAX_PARTICLES}).  Lower n_per_axis or raise the cap."
    )
    xs = np.linspace(-extent / 2, extent / 2, n_per_axis, dtype=np.float32)
    zs = np.linspace(-extent / 2, extent / 2, n_per_axis, dtype=np.float32)
    positions = np.zeros((_FLOOR_MAX_PARTICLES, 3), dtype=np.float32)
    masses    = np.zeros(_FLOOR_MAX_PARTICLES,       dtype=np.float32)
    for i, x in enumerate(xs):
        for j, z in enumerate(zs):
            idx = i * n_per_axis + j
            positions[idx, 0] = x
            positions[idx, 1] = floor_y
            positions[idx, 2] = z
            masses[idx]       = mass_per_particle
    _ti_floor_pos.from_numpy(positions)
    _ti_floor_pm.from_numpy(masses)
    _ti_clear_floor_gm_ul()
    _ti_scatter_floor_to_grid(n_floor)
    print(f"[floor] anchored {n_floor} static particles at y={floor_y}, "
          f"mass={mass_per_particle}/particle  (UL world-grid)")


@ti.kernel
def _ti_g2p_ul(start: ti.i32, end: ti.i32, dt: ti.f32,
               floor_damp: ti.f32, friction: ti.f32, use_floor: ti.i32):
    """Gather grid velocity + compute world-frame ∇v_p + multiplicative F update.
    Then apply identical floor/gripper/hammer contact as TL G2P so the
    boundary conditions are bit-identical between solvers."""
    eye = ti.Matrix.identity(ti.f32, 3)
    for p in range(start, end):
        new_v = ti.Vector([0.0, 0.0, 0.0])
        new_C = ti.Matrix.zero(ti.f32, 3, 3)
        grad_v = ti.Matrix.zero(ti.f32, 3, 3)   # TRUE velocity gradient  -  NOT the APIC C
        for k in ti.static(range(27)):
            idx = _ti_nidx_ul[p, k]
            w   = _ti_wip_ul[p, k]
            dw  = _ti_dwip_ul[p, k]
            dp  = _ti_dpos_ul[p, k]
            gv  = _ti_gv_ul[idx]
            new_v  += w * gv
            # APIC C  -  used only for next P2G momentum smoothing
            new_C  += (4.0 / (_UL_GRID_DX * _UL_GRID_DX)) * w * gv.outer_product(dp)
            # True ∇v for the multiplicative F update  -  uses ∇w gradient
            grad_v += gv.outer_product(dw)
        # APIC→PIC blend on C (Jiang 2015)  -  bleeds 5% of C per substep so a
        # body at rest doesn't store residual affine velocity that re-injects
        # as ghost momentum at the next P2G.  See solver_tl for the full
        # rationale.  The TRUE ∇v above (built from ∇w, not C) is unaffected,
        # so the multiplicative F update retains full accuracy.
        G._ti_C[p] = G.APIC_BLEND * new_C
        # MULTIPLICATIVE F update  -  the defining UL feature.  The
        # Stomakhin-style increment avoids inverted elastic states under
        # large melt/impact velocity gradients.
        G._ti_F[p] = _positive_det_F_increment(dt * grad_v) @ G._ti_F[p]

        # ----- Floor / gripper / hammer contact  -  IDENTICAL to TL G2P -----
        pos  = G._ti_pos[p]
        prox = ti.max(0.0, G.CONTACT_ZONE - pos[1])
        fx   = 0.0
        fz   = 0.0
        fy   = 0.0
        # Floor normal-direction is handled implicitly by the floor's anchor
        # mass in grid_update (the heavy floor dilutes body velocity at
        # contact nodes).  Here we apply Coulomb friction tangentially.
        # The "normal impulse" proxy in UL is gravity·dt (always pushing the
        # body into the floor at rest) plus any residual descending velocity
        # that survived the anchor dilution.  See solver_tl for the full
        # static-vs-kinetic decision rationale.
        if use_floor == 1 and prox > 0.0:
            v_n_total = ti.max(0.0, -new_v[1]) + ti.max(0.0, -G.GRAVITY_Y) * dt
            max_dv_t  = friction * v_n_total
            v_tx      = new_v[0]
            v_tz      = new_v[2]
            v_t_mag   = ti.sqrt(v_tx * v_tx + v_tz * v_tz)
            if v_t_mag > 1e-8:
                if v_t_mag <= max_dv_t:
                    new_v[0] = 0.0
                    new_v[2] = 0.0
                else:
                    scale = (v_t_mag - max_dv_t) / v_t_mag
                    new_v[0] = v_tx * scale
                    new_v[2] = v_tz * scale

        gy        = G._ti_gripper_y[None]
        grip_prox = ti.max(0.0, pos[1] - gy)
        if grip_prox > 0.0:
            fy -= G.FLOOR_K * grip_prox
            if new_v[1] > 0.0:
                fy -= floor_damp * new_v[1]
            fx -= G.FLOOR_FRIC * new_v[0]
            fz -= G.FLOOR_FRIC * new_v[2]

        bgy = G._ti_bottom_gripper_y[None]
        bottom_grip_prox = ti.max(0.0, bgy - pos[1])
        if bottom_grip_prox > 0.0:
            fy += G.FLOOR_K * bottom_grip_prox
            if new_v[1] < 0.0:
                fy -= floor_damp * new_v[1]
            fx -= G.FLOOR_FRIC * new_v[0]
            fz -= G.FLOOR_FRIC * new_v[2]

        lx = G._ti_side_plate_lx[None]
        left_prox = ti.max(0.0, lx - pos[0])
        if left_prox > 0.0:
            fx += G.FLOOR_K * left_prox
            if new_v[0] < 0.0:
                fx -= floor_damp * new_v[0]
            fy -= G.FLOOR_FRIC * new_v[1]
            fz -= G.FLOOR_FRIC * new_v[2]

        rx = G._ti_side_plate_rx[None]
        right_prox = ti.max(0.0, pos[0] - rx)
        if right_prox > 0.0:
            fx -= G.FLOOR_K * right_prox
            if new_v[0] > 0.0:
                fx -= floor_damp * new_v[0]
            fy -= G.FLOOR_FRIC * new_v[1]
            fz -= G.FLOOR_FRIC * new_v[2]

        hpos  = G._ti_hammer_pos[None]
        hdiff = pos - hpos
        hdist = hdiff.norm()
        if hdist < G.HAMMER_RADIUS and hdist > 1e-6:
            push   = hdiff / hdist
            pen    = G.HAMMER_RADIUS - hdist
            hvel   = G._ti_hammer_vel[None]
            rel_vn = (new_v - hvel).dot(push)
            h_damp = 2.0 * ti.sqrt(G.HAMMER_K * G._ti_m[p])
            cf     = ti.max(0.0, G.HAMMER_K * pen - h_damp * ti.min(rel_vn, 0.0))
            fx += cf * push[0]
            fy += cf * push[1]
            fz += cf * push[2]

        # Hand-skeleton contact (Tier 3)  -  see _ti_g2p_tl for the explanation.
        for bi in range(G.N_HAND_BONES):
            hand_id = bi // G.N_HAND_BONES_PER
            if G._ti_hand_active[hand_id] != 0:
                a_idx = G._ti_hand_bone_ab[bi, 0]
                b_idx = G._ti_hand_bone_ab[bi, 1]
                a_pt  = G._ti_hand_pts[a_idx]
                b_pt  = G._ti_hand_pts[b_idx]
                ab    = b_pt - a_pt
                ab2   = ab.dot(ab)
                tparam = 0.0
                if ab2 > 1e-12:
                    tparam = ti.min(1.0, ti.max(0.0, (pos - a_pt).dot(ab) / ab2))
                closest = a_pt + tparam * ab
                diff    = pos - closest
                bdist   = diff.norm()
                if bdist < G.HAND_RADIUS and bdist > 1e-6:
                    bpush  = diff / bdist
                    bpen   = G.HAND_RADIUS - bdist
                    bvel   = (1.0 - tparam) * G._ti_hand_pts_v[a_idx] + tparam * G._ti_hand_pts_v[b_idx]
                    brelvn = (new_v - bvel).dot(bpush)
                    bdamp  = 2.0 * ti.sqrt(G.HAND_K * G._ti_m[p])
                    bcf    = ti.max(0.0, G.HAND_K * bpen - bdamp * ti.min(brelvn, 0.0))
                    fx    += bcf * bpush[0]
                    fy    += bcf * bpush[1]
                    fz    += bcf * bpush[2]

        inv_m = dt / G._ti_m[p]
        decay = ti.exp(-G._ti_damping[p] * dt)
        # Material sound speed c = √(stiffness/ρ)  -  the real velocity limit.
        # Above c, stress can't propagate fast enough to hold the body.
        # Use the LARGER of shear (μ) and bulk (κ) stiffness so molten
        # particles (μ=0, κ=molten_kappa) are bounded by their pressure-wave
        # speed √(κ/ρ) rather than √(0/ρ)=0.  Without this a fluid particle's
        # velocity is clamped to zero every substep and the melt cannot flow.
        rho_p  = ti.max(G._ti_m[p] / ti.max(G._ti_v0[p], 1e-12), 1e-6)
        stiff  = ti.max(G._ti_mu[p], G._ti_kappa_p[p])
        v_cap  = G.MAX_SPEED                       # CFL fallback for stiffness≈0
        if stiff > 1e-3:
            v_cap = ti.min(ti.sqrt(stiff / rho_p), G.MAX_SPEED)
        fv = ti.Vector([
            (new_v[0] + fx * inv_m) * decay,
            (new_v[1] + fy * inv_m) * decay,
            (new_v[2] + fz * inv_m) * decay,
        ])
        fv_norm = fv.norm()
        if fv_norm > v_cap:
            fv = fv * (v_cap / ti.max(fv_norm, 1e-8))
        # Settle-decay applied independently per-axis  -  see solver_tl for the
        # rationale.  Kills sub-threshold residual creep on the floor.
        if G._ti_m[p] > 1e-4 and prox > 0.0:
            decay_settle = ti.exp(-G.FLOOR_SETTLE_ALPHA * dt)
            if ti.abs(fv[0]) < G.FLOOR_SETTLE_SPEED:
                fv[0] = fv[0] * decay_settle
            if ti.abs(fv[1]) < G.FLOOR_SETTLE_SPEED:
                fv[1] = fv[1] * decay_settle
            if ti.abs(fv[2]) < G.FLOOR_SETTLE_SPEED:
                fv[2] = fv[2] * decay_settle
        G._ti_vel[p] = fv
        G._ti_pos[p] = pos + fv * dt

        # Positional safety net  -  completes the separating-floor constraint
        # in case gripper/hammer/hand forces pushed a particle below floor
        # within this substep.  Don't touch vy.
        if use_floor == 1 and G._ti_pos[p][1] < G.CONTACT_ZONE:
            G._ti_pos[p][1] = G.CONTACT_ZONE


# ============================================================
# Public API
# ============================================================

def substep(start: int, end: int, dt: float, kappa: float, mu_scale: float,
            model: int, floor_damp: float, friction: float,
            project: bool = False, fluid_rho: float = 1000.0,
            project_iters: int = 80, use_floor: bool = True,
            physgaussian_ficus_bc: bool = False,
            bounding_box_bc: bool = False,
            grid_bc_fn=None) -> None:
    """One UL-MLS-MPM substep for particles [start, end).
        clear → build_world_grid → p2g → grid_update → floor_bc
        → [Chorin projection for fluid cells] → g2p
    project=True enables the incompressible pressure projection, which enforces
    ∇·v=0 in fluid cells via red-black Gauss-Seidel.  Makes molten material
    nearly incompressible so the solid chunk floats on top of the melt."""
    _ti_clear_grid_ul()
    _ti_build_world_grid_ul(start, end)
    _ti_p2g_ul(start, end, kappa, mu_scale, model)
    _ti_grid_update_ul(dt)
    if bounding_box_bc:
        _ti_apply_physgaussian_bounding_box_bc()
    elif physgaussian_ficus_bc:
        if G.PHYSGAUSSIAN_FICUS_DROP:
            _ti_apply_physgaussian_bounding_box_bc()
        else:
            _ti_apply_physgaussian_ficus_cuboid_bc()
    elif use_floor:
        _ti_apply_floor_bc_ul(G.CONTACT_ZONE)
    if grid_bc_fn is not None:
        # Caller-supplied grid-level boundary conditions (for example the
        # benchmark harness's prescribed-displacement slabs). Particle-level
        # pins alone cannot express a displacement BC in MPM: a one-layer
        # pinned band's velocity is mass-diluted by the B-spline stencil and
        # transmits almost no force, so the constraint must also be imposed
        # on the grid velocities, exactly like the floor BC above.
        grid_bc_fn()
    if project:
        pressure_project(start, end, dt, fluid_rho, n_iters=project_iters)
    _ti_g2p_ul(start, end, dt, floor_damp, friction, 1 if use_floor else 0)


def substep_multi(entities, dt: float) -> None:
    """Multi-entity substep on a SHARED world grid.  All entities scatter to
    the same grid before the grid update, then all gather after.  This is
    how body-body collision works in UL  -  the mass-weighted grid velocity
    average naturally couples overlapping bodies.

    `entities` is a list of dicts with keys:
        start, end, kappa, mu_scale, model, floor_damp, friction
    """
    _ti_clear_grid_ul()
    for e in entities:
        _ti_build_world_grid_ul(e["start"], e["end"])
    for e in entities:
        _ti_p2g_ul(e["start"], e["end"], e["kappa"], e["mu_scale"], e["model"])
    _ti_grid_update_ul(dt)
    _ti_apply_floor_bc_ul(G.CONTACT_ZONE)
    for e in entities:
        _ti_g2p_ul(e["start"], e["end"], dt, e["floor_damp"], e["friction"], 1)


# ============================================================
# Thermodynamics  -  per-frame grid heat diffusion (Stomakhin 2014)
# ============================================================
# Heat runs ONCE per display frame (thermal timescale ≫ mechanical substep).
# Reuses the world grid: scatter particle T → grid, explicit Laplacian
# diffusion with a floor Dirichlet source, gather grid T → particle.
# Pure one-way coupling: temperature drives material params, not vice-versa.

@ti.kernel
def _ti_heat_clear_ul():
    for i in range(_MAX_GRID_UL):
        _ti_gTmass_ul[i] = 0.0
        _ti_gTm_ul[i]    = 0.0
        _ti_gT_ul[i]     = 0.0


@ti.kernel
def _ti_heat_scatter_ul(start: ti.i32, end: ti.i32):
    """Scatter w·m and w·m·T to the grid using the current UL stencil."""
    for p in range(start, end):
        m = G._ti_m[p]
        T = G._ti_temp[p]
        for k in ti.static(range(27)):
            idx = _ti_nidx_ul[p, k]
            w   = _ti_wip_ul[p, k]
            ti.atomic_add(_ti_gTmass_ul[idx], w * m)
            ti.atomic_add(_ti_gTm_ul[idx],    w * m * T)


@ti.kernel
def _ti_heat_normalize_ul():
    for i in range(_MAX_GRID_UL):
        if _ti_gTmass_ul[i] > 1e-12:
            _ti_gT_ul[i] = _ti_gTm_ul[i] / _ti_gTmass_ul[i]
        else:
            _ti_gT_ul[i] = 0.0


@ti.kernel
def _ti_heat_source_ul(floor_y: ti.f32, band_dx: ti.f32, source_temp: ti.f32):
    """Dirichlet heat source: clamp grid temperature in the floor band to
    source_temp, for cells that carry mass (the body touching the floor)."""
    origin_y = float(_UL_GRID_ORIGIN[1])
    dx       = _UL_GRID_DX
    band_top = floor_y + band_dx * dx
    for i in range(_MAX_GRID_UL):
        if _ti_gTmass_ul[i] > 1e-12:
            iy = (i // _UL_GRID_N) % _UL_GRID_N
            node_y = origin_y + ti.cast(iy, ti.f32) * dx
            if node_y <= band_top:
                _ti_gT_ul[i] = source_temp


@ti.kernel
def _ti_heat_diffuse_ul(alpha: ti.f32):
    """One explicit Laplacian step: T_new[i] = T[i] + α·Σ_neighbours(T[j]−T[i]).
    Only diffuses among mass-bearing cells (heat doesn't leak into vacuum).
    Writes _ti_gT_new_ul (double-buffer)."""
    N  = _UL_GRID_N
    NN = N * N
    for i in range(_MAX_GRID_UL):
        if _ti_gTmass_ul[i] > 1e-12:
            iz = i % N
            iy = (i // N) % N
            ix = i // NN
            Ti = _ti_gT_ul[i]
            lap = 0.0
            # 6-neighbour stencil; only count neighbours that carry mass
            for axis in ti.static(range(3)):
                for s in ti.static((-1, 1)):
                    jx, jy, jz = ix, iy, iz
                    if axis == 0:   jx = ix + s
                    elif axis == 1: jy = iy + s
                    else:           jz = iz + s
                    if 0 <= jx < N and 0 <= jy < N and 0 <= jz < N:
                        j = jx * NN + jy * N + jz
                        if _ti_gTmass_ul[j] > 1e-12:
                            lap += (_ti_gT_ul[j] - Ti)
            _ti_gT_new_ul[i] = Ti + alpha * lap
        else:
            _ti_gT_new_ul[i] = 0.0


@ti.kernel
def _ti_heat_swap_ul():
    for i in range(_MAX_GRID_UL):
        _ti_gT_ul[i] = _ti_gT_new_ul[i]


@ti.kernel
def _ti_heat_gather_ul(start: ti.i32, end: ti.i32):
    """Gather grid temperature back to particles (mass-weighted).

    Uses w·m_grid as the gather weight so surface particles  -  whose stencil
    spans empty nodes  -  are not pulled toward zero.  Equivalent to computing
    Σ(w·gTm) / Σ(w·gTmass) since T_grid = gTm/gTmass, which conserves the
    total mass-weighted heat Σ(m_p·T_p) across the P2G→G2P round-trip."""
    for p in range(start, end):
        T_new = 0.0
        msum  = 0.0
        for k in ti.static(range(27)):
            idx = _ti_nidx_ul[p, k]
            w   = _ti_wip_ul[p, k]
            mg  = _ti_gTmass_ul[idx]
            if mg > 1e-12:
                T_new += w * mg * _ti_gT_ul[idx]
                msum  += w * mg
        if msum > 1e-12:
            G._ti_temp[p] = T_new / msum


# ============================================================
# Chorin pressure projection  -  incompressible fluid on MPM grid
# ============================================================
# Sequence per substep (fluid cells only):
#   scatter fluid mass → compute ∇·v → Jacobi pressure solve → project v
#
# Cell types (per grid node):
#   fluid  : _ti_gfluid_m_ul > threshold (molten-particle mass present)
#   solid  : has body/floor mass but no fluid mass (Neumann ∂p/∂n=0)
#   vacuum : no mass at all (Dirichlet p=0, free surface)
#
# Jacobi vs red-black G-S: Jacobi reads p_old → writes p_new for ALL fluid
# cells simultaneously.  Slower convergence but no race conditions, no
# colour-update ordering issues, and deterministically stable.

_PROJ_P_MAX = 3.0e5    # pressure clamp: ±300 kPa (prevents runaway on first iter)
_PROJ_DV_MAX = 1.0     # max velocity correction per substep, m/s


@ti.kernel
def _ti_proj_clear_ul():
    for i in range(_MAX_GRID_UL):
        _ti_gfluid_m_ul[i] = 0.0
        _ti_gp_ul[i]       = 0.0
        _ti_gp_new_ul[i]   = 0.0
        _ti_gdiv_ul[i]     = 0.0


@ti.kernel
def _ti_proj_fluid_mass_ul(start: ti.i32, end: ti.i32):
    """Scatter mass from fluid-phase (phase==1) particles → _ti_gfluid_m_ul."""
    for p in range(start, end):
        if G._ti_phase[p] == 1:
            m = G._ti_m[p]
            for k in ti.static(range(27)):
                ti.atomic_add(_ti_gfluid_m_ul[_ti_nidx_ul[p, k]],
                              _ti_wip_ul[p, k] * m)


@ti.kernel
def _ti_proj_divergence_ul():
    """Central-difference ∇·v at fluid cells using the current grid velocity."""
    N  = _UL_GRID_N
    NN = N * N
    dx = _UL_GRID_DX
    for i in range(_MAX_GRID_UL):
        if _ti_gfluid_m_ul[i] > 1e-12:
            iz = i % N
            iy = (i // N) % N
            ix = i // NN
            v  = _ti_gv_ul[i]
            div = 0.0
            # x
            vxp = _ti_gv_ul[(min(ix+1,N-1))*NN + iy*N + iz][0]
            vxm = _ti_gv_ul[(max(ix-1,0  ))*NN + iy*N + iz][0]
            denom_x = (1 if ix > 0 and ix < N-1 else 1) * 2  # always 2*dx for clamped
            if ix == 0:    vxm = v[0]   # Neumann at left wall
            if ix == N-1:  vxp = v[0]   # Neumann at right wall
            div += (vxp - vxm) / (2.0 * dx)
            # y
            vyp = _ti_gv_ul[ix*NN + (min(iy+1,N-1))*N + iz][1]
            vym = _ti_gv_ul[ix*NN + (max(iy-1,0  ))*N + iz][1]
            if iy == 0:    vym = v[1]
            if iy == N-1:  vyp = v[1]
            div += (vyp - vym) / (2.0 * dx)
            # z
            vzp = _ti_gv_ul[ix*NN + iy*N + min(iz+1,N-1)][2]
            vzm = _ti_gv_ul[ix*NN + iy*N + max(iz-1,0  )][2]
            if iz == 0:    vzm = v[2]
            if iz == N-1:  vzp = v[2]
            div += (vzp - vzm) / (2.0 * dx)
            _ti_gdiv_ul[i] = div


@ti.kernel
def _ti_proj_jacobi_ul(dt: ti.f32, rho: ti.f32):
    """One Jacobi sweep: reads _ti_gp_ul, writes _ti_gp_new_ul.
    Poisson equation:  ∇²p = (ρ/dt) · ∇·v
    Discretised:  p[i] = (Σ_{fluid j} p[j] − dx²·ρ/dt·div[i]) / n_eff
    BCs: p=0 at vacuum (free surface), ∂p/∂n=0 at solid (not counted)."""
    N  = _UL_GRID_N
    NN = N * N
    dx = _UL_GRID_DX
    p_max = ti.static(_PROJ_P_MAX)
    for i in range(_MAX_GRID_UL):
        if _ti_gfluid_m_ul[i] > 1e-12:
            iz = i % N
            iy = (i // N) % N
            ix = i // NN
            p_sum = 0.0
            n_eff = 0.0
            # +x neighbour
            if ix + 1 < N:
                j = (ix+1)*NN + iy*N + iz
                if _ti_gfluid_m_ul[j] > 1e-12:
                    p_sum += _ti_gp_ul[j]; n_eff += 1.0
                elif _ti_gm_ul[j] + _ti_floor_gm_ul[j] < 1e-12:
                    n_eff += 1.0  # vacuum → p=0 Dirichlet
            # -x neighbour
            if ix - 1 >= 0:
                j = (ix-1)*NN + iy*N + iz
                if _ti_gfluid_m_ul[j] > 1e-12:
                    p_sum += _ti_gp_ul[j]; n_eff += 1.0
                elif _ti_gm_ul[j] + _ti_floor_gm_ul[j] < 1e-12:
                    n_eff += 1.0
            # +y neighbour
            if iy + 1 < N:
                j = ix*NN + (iy+1)*N + iz
                if _ti_gfluid_m_ul[j] > 1e-12:
                    p_sum += _ti_gp_ul[j]; n_eff += 1.0
                elif _ti_gm_ul[j] + _ti_floor_gm_ul[j] < 1e-12:
                    n_eff += 1.0
            # -y neighbour
            if iy - 1 >= 0:
                j = ix*NN + (iy-1)*N + iz
                if _ti_gfluid_m_ul[j] > 1e-12:
                    p_sum += _ti_gp_ul[j]; n_eff += 1.0
                elif _ti_gm_ul[j] + _ti_floor_gm_ul[j] < 1e-12:
                    n_eff += 1.0
            # +z neighbour
            if iz + 1 < N:
                j = ix*NN + iy*N + (iz+1)
                if _ti_gfluid_m_ul[j] > 1e-12:
                    p_sum += _ti_gp_ul[j]; n_eff += 1.0
                elif _ti_gm_ul[j] + _ti_floor_gm_ul[j] < 1e-12:
                    n_eff += 1.0
            # -z neighbour
            if iz - 1 >= 0:
                j = ix*NN + iy*N + (iz-1)
                if _ti_gfluid_m_ul[j] > 1e-12:
                    p_sum += _ti_gp_ul[j]; n_eff += 1.0
                elif _ti_gm_ul[j] + _ti_floor_gm_ul[j] < 1e-12:
                    n_eff += 1.0
            if n_eff > 1e-6:
                rhs = (rho / dt) * _ti_gdiv_ul[i]
                p_raw = (p_sum - dx * dx * rhs) / n_eff
                # Clamp: prevents runaway in early iterations / near-degenerate cells
                _ti_gp_new_ul[i] = ti.min(ti.max(p_raw, -p_max), p_max)
            else:
                _ti_gp_new_ul[i] = 0.0


@ti.kernel
def _ti_proj_swap_ul():
    for i in range(_MAX_GRID_UL):
        _ti_gp_ul[i] = _ti_gp_new_ul[i]


@ti.kernel
def _ti_proj_apply_ul(dt: ti.f32, rho: ti.f32):
    """Subtract ∇p/ρ from grid velocity at fluid cells (central difference).
    Correction is clamped to _PROJ_DV_MAX to prevent instability."""
    N  = _UL_GRID_N
    NN = N * N
    dx = _UL_GRID_DX
    dv_max = ti.static(_PROJ_DV_MAX)
    for i in range(_MAX_GRID_UL):
        if _ti_gfluid_m_ul[i] > 1e-12:
            iz = i % N
            iy = (i // N) % N
            ix = i // NN
            # Central difference pressure gradient; clamp at boundaries
            pp_x = _ti_gp_ul[(min(ix+1,N-1))*NN + iy*N + iz]
            pm_x = _ti_gp_ul[(max(ix-1,0  ))*NN + iy*N + iz]
            pp_y = _ti_gp_ul[ix*NN + (min(iy+1,N-1))*N + iz]
            pm_y = _ti_gp_ul[ix*NN + (max(iy-1,0  ))*N + iz]
            pp_z = _ti_gp_ul[ix*NN + iy*N + min(iz+1,N-1)]
            pm_z = _ti_gp_ul[ix*NN + iy*N + max(iz-1,0  )]
            correction = ti.Vector([
                (pp_x - pm_x) / (2.0 * dx),
                (pp_y - pm_y) / (2.0 * dx),
                (pp_z - pm_z) / (2.0 * dx),
            ]) * (dt / rho)
            # Clamp magnitude to prevent large corrections blowing up velocities
            cn = correction.norm()
            if cn > dv_max:
                correction = correction * (dv_max / cn)
            _ti_gv_ul[i] -= correction


def pressure_project(start: int, end: int, dt: float, rho: float,
                     n_iters: int = 40) -> None:
    """Chorin pressure projection: enforces ∇·v≈0 in fluid cells.
        clear → scatter fluid mass → divergence → Jacobi×n_iters → project
    Uses Jacobi iteration (read p_old, write p_new) with pressure and velocity
    correction clamping for stability in the mixed solid/fluid MPM regime."""
    _ti_proj_clear_ul()
    _ti_proj_fluid_mass_ul(start, end)
    _ti_proj_divergence_ul()
    for _ in range(n_iters):
        _ti_proj_jacobi_ul(dt, rho)
        _ti_proj_swap_ul()
    _ti_proj_apply_ul(dt, rho)


def heat_step(start: int, end: int, dt: float,
              conductivity: float, heat_capacity: float, density: float,
              floor_y: float, source_temp: float,
              n_iters: int = 8, band_dx: float = 2.0) -> None:
    """One per-frame heat solve over particles [start, end).
        clear → build stencil → scatter T → normalize → (source + diffuse)×N → gather
    alpha is the explicit-diffusion coefficient, CFL-clamped to <0.2 for stability."""
    dx = _UL_GRID_DX
    alpha = conductivity * dt / (max(density * heat_capacity, 1e-6) * dx * dx)
    alpha = min(alpha, 0.16)   # explicit stability (6-neighbour bound is 1/6)
    _ti_heat_clear_ul()
    _ti_build_world_grid_ul(start, end)
    _ti_heat_scatter_ul(start, end)
    _ti_heat_normalize_ul()
    for _ in range(n_iters):
        _ti_heat_source_ul(floor_y, band_dx, source_temp)
        _ti_heat_diffuse_ul(alpha)
        _ti_heat_swap_ul()
    _ti_heat_source_ul(floor_y, band_dx, source_temp)
    _ti_heat_gather_ul(start, end)


def heat_step_physical(start: int, end: int, dt: float,
                       conductivity: float, heat_capacity: float,
                       density: float, floor_y: float, source_temp: float,
                       band_dx: float = 1.0) -> None:
    """SI-calibrated per-frame heat solve (GaussianBench A4 path).

    heat_step above computes the physically correct explicit coefficient
    for the FULL frame dt and then applies it n_iters times, so its
    effective diffusivity is n_iters times the physical value (or, when
    the 0.16 clamp engages, determined by the iteration count alone).
    That is fine for look-tuned demos but useless against an analytical
    reference. This variant subdivides the frame instead: the fewest
    iterations satisfying the explicit stability bound, each advancing
        alpha_iter = alpha_SI * (dt / n) / dx^2,
    so the total diffusion over the frame equals k/(rho c) exactly.
    Temperatures are whatever unit the caller feeds (Kelvin works: every
    operation here is affine in T). heat_step is left untouched.
    """
    dx = _UL_GRID_DX
    alpha_si = conductivity / max(density * heat_capacity, 1e-9)
    alpha_frame = alpha_si * dt / (dx * dx)
    n_iters = max(1, int(np.ceil(alpha_frame / 0.16)))
    alpha_iter = alpha_frame / n_iters
    _ti_heat_clear_ul()
    _ti_build_world_grid_ul(start, end)
    _ti_heat_scatter_ul(start, end)
    _ti_heat_normalize_ul()
    for _ in range(n_iters):
        _ti_heat_source_ul(floor_y, band_dx, source_temp)
        _ti_heat_diffuse_ul(alpha_iter)
        _ti_heat_swap_ul()
    _ti_heat_source_ul(floor_y, band_dx, source_temp)
    _ti_heat_gather_ul(start, end)
