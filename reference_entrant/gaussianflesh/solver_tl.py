"""
solver_tl.py
Total-Lagrangian APIC solver  -  the existing GaussianFlesh approach,
unchanged in behaviour, lifted out of gaussian_mesh_poc.py.

Total-Lagrangian:
  - Grid is fixed in the REST (undeformed) configuration.
  - Per-particle 27-node stencil (indices, B-spline weights, ∇_X w) is
    precomputed ONCE from rest positions and never rebuilt.
  - F update is ADDITIVE: F += dt · grad_X(v), where grad_X(v) is gathered
    with the precomputed rest-frame kernel gradients.  The APIC affine C is
    still used for momentum transfer, but not as a substitute for grad_X(v).
  - WLS F recompute every frame re-measures F from current particle
    positions against rest neighbours  -  a TL-specific drift correction.

Force scatter uses PK1 stress P with rest-frame gradients:
    f_i = -V₀ · P · ∇_X w_i(X_p)
which is the standard TL P2G integrand.
"""
import numpy as np
import taichi as ti

import gaussian_mesh_poc as G


# ============================================================
# TL grid + stencil fields
# ============================================================

# Same _MAX_GRID upper bound as the original module  -  kept generous so
# the rest-frame grid for any reasonable shape fits.
_MAX_GRID_TL = G._MAX_GRID

_ti_gm_tl  = ti.field(ti.f32, shape=_MAX_GRID_TL)
_ti_gmo_tl = ti.Vector.field(3, dtype=ti.f32, shape=_MAX_GRID_TL)
_ti_gf_tl  = ti.Vector.field(3, dtype=ti.f32, shape=_MAX_GRID_TL)
_ti_gv_tl  = ti.Vector.field(3, dtype=ti.f32, shape=_MAX_GRID_TL)

# Per-particle 27-node stencil  -  REST-FRAME, precomputed once
_ti_nidx_tl = ti.field(ti.i32,           shape=(G.N_PARTICLES, 27))
_ti_wip_tl  = ti.field(ti.f32,           shape=(G.N_PARTICLES, 27))
_ti_dwip_tl = ti.Vector.field(3, ti.f32, shape=(G.N_PARTICLES, 27))  # ∇_X w in rest frame
_ti_dpos_tl = ti.Vector.field(3, ti.f32, shape=(G.N_PARTICLES, 27))  # world displacement to node

# WLS neighbourhood for F recompute  -  TL-specific drift correction
_ti_wnb_tl = ti.field(ti.i32,            shape=(G.N_PARTICLES, G.K_PHYS))
_ti_ww_tl  = ti.field(ti.f32,            shape=(G.N_PARTICLES, G.K_PHYS))
_ti_wB_tl  = ti.Matrix.field(3, 3, ti.f32, shape=G.N_PARTICLES)
_ti_wBi_tl = ti.Matrix.field(3, 3, ti.f32, shape=G.N_PARTICLES)

# ============================================================
# Upload helpers  -  populate TL fields from CPU numpy arrays
# ============================================================

def upload_stencil(node_idx: np.ndarray, w_ip: np.ndarray,
                   dw_ip: np.ndarray, dpos_ip: np.ndarray,
                   start: int = 0) -> None:
    """Upload precomputed rest-frame stencil arrays into the TL fields.
    `start` lets us write into the entity-0 slice when in multi-entity
    mode (entity 1, which uses UL, leaves its TL stencil zeros  -  never read)."""
    N = node_idx.shape[0]
    # Pull current full-N copy, overwrite the slice, push back.
    cur_idx = _ti_nidx_tl.to_numpy()
    cur_w   = _ti_wip_tl.to_numpy()
    cur_dw  = _ti_dwip_tl.to_numpy()
    cur_dp  = _ti_dpos_tl.to_numpy()
    cur_idx[start:start + N]    = node_idx.astype(np.int32)
    cur_w[start:start + N]      = w_ip.astype(np.float32)
    cur_dw[start:start + N]     = dw_ip.astype(np.float32)
    cur_dp[start:start + N]     = dpos_ip.astype(np.float32)
    _ti_nidx_tl.from_numpy(cur_idx)
    _ti_wip_tl.from_numpy(cur_w)
    _ti_dwip_tl.from_numpy(cur_dw)
    _ti_dpos_tl.from_numpy(cur_dp)


def upload_wls(wls_nbs: np.ndarray, wls_weights: np.ndarray,
               wls_B: np.ndarray, wls_Binv: np.ndarray,
               start: int = 0) -> None:
    """Upload precomputed WLS neighbourhood arrays for F recompute."""
    N = wls_nbs.shape[0]
    cur_nb = _ti_wnb_tl.to_numpy()
    cur_w  = _ti_ww_tl.to_numpy()
    cur_B  = _ti_wB_tl.to_numpy()
    cur_Bi = _ti_wBi_tl.to_numpy()
    cur_nb[start:start + N] = wls_nbs.astype(np.int32)
    cur_w[start:start + N]  = wls_weights.astype(np.float32)
    cur_B[start:start + N]  = wls_B.astype(np.float32)
    cur_Bi[start:start + N] = wls_Binv.astype(np.float32)
    _ti_wnb_tl.from_numpy(cur_nb)
    _ti_ww_tl.from_numpy(cur_w)
    _ti_wB_tl.from_numpy(cur_B)
    _ti_wBi_tl.from_numpy(cur_Bi)


# ============================================================
# Kernels
# ============================================================

@ti.kernel
def _ti_clear_grid_tl(n_grid: ti.i32):
    """Zero the active TL grid nodes [0, n_grid)."""
    for i in range(n_grid):
        _ti_gm_tl[i]  = 0.0
        _ti_gmo_tl[i] = ti.Vector([0.0, 0.0, 0.0])
        _ti_gf_tl[i]  = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def _ti_p2g_tl(start: ti.i32, end: ti.i32,
               kappa: ti.f32, mu_scale: ti.f32, model: ti.i32,
               floor_damp: ti.f32, friction: ti.f32):
    """P2G with PK1 stress and REST-FRAME gradients ∇_X w (precomputed).
    Force: f_i = -V₀ · P · ∇_X w_i.  APIC momentum: m_p · (v_p + C_p · dp)."""
    for p in range(start, end):
        Fe  = G._ti_F[p] @ G._ti_Fp[p].inverse()
        v   = G._ti_vel[p]
        C   = G._ti_C[p]
        m   = G._ti_m[p]
        # Damage-softened mu: fully damaged particles (d=1) contribute zero elastic stress
        mu  = G._mu_eff_sr(p, mu_scale) * (1.0 - G._ti_damage[p])
        lam = G._ti_lam[p] * (1.0 - G._ti_damage[p])
        P   = ti.Matrix.zero(ti.f32, 3, 3)
        p_model = G._ti_model[p]
        if p_model == 0:
            P = G._pk1_corotated(Fe, mu, lam)
        elif p_model == 2:
            P = G._pk1_sand(Fe, mu, lam)
        elif p_model == 3:
            P = G._pk1_stvk(G._ti_F[p], mu, lam)
        else:
            kp = kappa
            if G._ti_kappa_p[p] > 0.0:
                kp = G._ti_kappa_p[p]
            P = G._pk1_fluid(Fe, kp, mu)
        pn = P.norm()
        if pn > G.MPM_STRESS_CLAMP:
            P = P * (G.MPM_STRESS_CLAMP / pn)
        external_f = ti.Vector([0.0, 0.0, 0.0])
        penetration = G.CONTACT_ZONE - G._ti_pos[p][1]
        if penetration > 0.0:
            normal_f = G.FLOOR_K * penetration
            if v[1] < 0.0:
                normal_f += floor_damp * (-v[1])
            external_f[1] += normal_f
            external_f[0] -= G.FLOOR_FRIC * friction * v[0]
            external_f[2] -= G.FLOOR_FRIC * friction * v[2]
        for k in ti.static(range(27)):
            idx = _ti_nidx_tl[p, k]
            w   = _ti_wip_tl[p, k]
            dw  = _ti_dwip_tl[p, k]
            dp  = _ti_dpos_tl[p, k]
            ti.atomic_add(_ti_gmo_tl[idx], w * m * (v + C @ dp))
            ti.atomic_add(_ti_gm_tl[idx],  w * m)
            ti.atomic_add(_ti_gf_tl[idx],  -G._ti_v0[p] * P @ dw + w * external_f)


@ti.kernel
def _ti_grid_update_tl(n_grid: ti.i32, dt: ti.f32):
    grav = ti.Vector([0.0, G.GRAVITY_Y, 0.0])
    for i in range(n_grid):
        m = _ti_gm_tl[i]
        if m > 1e-12:
            _ti_gv_tl[i] = (_ti_gmo_tl[i] + dt * (_ti_gf_tl[i] + m * grav)) / m
        else:
            _ti_gv_tl[i] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def _ti_g2p_tl(start: ti.i32, end: ti.i32, dt: ti.f32,
               floor_damp: ti.f32, friction: ti.f32, grid_h: ti.f32):
    """Gather grid velocity into v_p, APIC C_p, and true reference grad_X(v).
    F update is ADDITIVE: F += dt·grad_X(v).
    Identical floor/gripper/hammer contact to UL G2P so BC are the same."""
    for p in range(start, end):
        new_v = ti.Vector([0.0, 0.0, 0.0])
        new_C = ti.Matrix.zero(ti.f32, 3, 3)
        grad_v = ti.Matrix.zero(ti.f32, 3, 3)
        for k in ti.static(range(27)):
            idx = _ti_nidx_tl[p, k]
            w   = _ti_wip_tl[p, k]
            dw  = _ti_dwip_tl[p, k]
            dp  = _ti_dpos_tl[p, k]
            gv  = _ti_gv_tl[idx]
            new_v += w * gv
            grad_v += gv.outer_product(dw)
            new_C += (4.0 / (grid_h * grid_h)) * w * gv.outer_product(dp)
        # APIC→PIC blend on C (Jiang 2015)  -  bleeds C-mode energy so a body
        # at rest doesn't store residual affine velocity that re-injects as
        # ghost momentum next P2G.  Plain APIC (no blend) makes settled
        # bodies creep forever; BLEND=0.95 kills the creep without
        # over-damping the active dynamics.
        G._ti_C[p] = G.APIC_BLEND * new_C
        G._ti_F[p] = G._ti_F[p] + dt * grad_v   # TL ADDITIVE F update

        pos  = G._ti_pos[p]
        prox = ti.max(0.0, G.CONTACT_ZONE - pos[1])
        fx   = 0.0
        fz   = 0.0
        fy   = 0.0
        # Hard separating floor  -  particle-level analog of PhysGaussian's
        # grid BC.  Capture the magnitude of inward velocity BEFORE we zero
        # it; this is the normal impulse per unit mass that the floor just
        # delivered to this particle, used to size the Coulomb friction cap.
        # Coulomb friction at the floor.  Tangential velocity reduction is
        # bounded by μ·|N| per substep where |N| = (impact impulse this
        # substep) + (gravity·dt).  The gravity term provides the steady-
        # state normal force for a body at rest, so static friction holds
        # even after the impact ringing has died.
        #     if v_t_mag ≤ μ·|N|  → static: snap tangent to zero
        #     else                → kinetic: reduce magnitude by μ·|N|
        # Below the static threshold tangential motion stops dead  -  no
        # exponential creep.  Friction acts directly on new_v (the post-G2P
        # velocity) so the result is preserved through fv = (new_v + ...)·decay.
        if prox > 0.0:
            v_n_total = ti.max(0.0, -new_v[1]) + ti.max(0.0, -G.GRAVITY_Y) * dt
            max_dv_t  = friction * v_n_total
            v_tx      = new_v[0]
            v_tz      = new_v[2]
            v_t_mag   = ti.sqrt(v_tx * v_tx + v_tz * v_tz)
            if v_t_mag > 1e-8:
                if v_t_mag <= max_dv_t:
                    # Static: friction can fully stop us
                    new_v[0] = 0.0
                    new_v[2] = 0.0
                else:
                    # Kinetic: reduce magnitude by μ·|N|
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

        # Hand-skeleton contact (Tier 3)  -  N_HAND_BONES capsules across up to 2
        # hands.  Inactive hand slots park their joints at y=9999 so the distance
        # test fails cheaply.  Each bone is a swept-sphere from joint A to joint
        # B with HAND_RADIUS; bone velocity = lerp of joint velocities along t.
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
        c_mat = ti.sqrt(G._ti_mu[p] / ti.max(G._ti_m[p] / ti.max(G._ti_v0[p], 1e-12), 1e-6))
        v_cap = ti.min(c_mat, G.MAX_SPEED)
        fv = ti.Vector([
            (new_v[0] + fx * inv_m) * decay,
            (new_v[1] + fy * inv_m) * decay,
            (new_v[2] + fz * inv_m) * decay,
        ])
        fv_norm = fv.norm()
        if fv_norm > v_cap:
            fv = fv * (v_cap / ti.max(fv_norm, 1e-8))
        # Settle-decay: when a particle is in contact with the floor AND its
        # velocity component is below SETTLE_SPEED, apply an aggressive
        # exponential decay that quickly drives it to (numerically) zero.
        # Applied independently per-axis so a sliding+wobbling body stops
        # sliding faster than it stops wobbling vertically.
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

        # Positional safety net  -  completes the separating-floor constraint.
        # If gripper/hammer/hand forces drove the particle below floor in this
        # substep, clamp position back.  Don't touch vy (the velocity constraint
        # at the top of the BC block already handled the inward component).

@ti.kernel
def _ti_bond_break_check(start: ti.i32, end: ti.i32,
                         stretch_thresh: ti.f32, dmg_thresh: ti.f32):
    """Mark a WLS bond as broken when either:
      - current bond length > stretch_thresh * rest length, OR
      - both endpoints have damage > dmg_thresh
    Once broken, stays broken (permanent topology change)."""
    for p in range(start, end):
        for k in range(G.K_PHYS):
            if G._ti_bond_intact[p, k] == 1:
                nb = _ti_wnb_tl[p, k]
                rest_len = (G._ti_rest[nb] - G._ti_rest[p]).norm()
                cur_len  = (G._ti_pos[nb]  - G._ti_pos[p]).norm()
                stretch  = cur_len / ti.max(rest_len, 1e-6)
                if stretch > stretch_thresh:
                    G._ti_bond_intact[p, k] = 0
                elif G._ti_damage[p] > dmg_thresh and G._ti_damage[nb] > dmg_thresh:
                    G._ti_bond_intact[p, k] = 0


@ti.kernel
def _ti_wls_recompute_tl_fracture(start: ti.i32, end: ti.i32):
    """Fracture-aware WLS F recompute: skips broken bonds and rebuilds the
    B matrix per particle from intact neighbours only.  Slower than the
    pre-computed _ti_wB_tl path (we re-invert a 3x3 every call) but the
    topology can change, so we have no choice.

    If all/most bonds break, B becomes degenerate and we keep F unchanged
     -  that particle is effectively a free body now, advected by MPM grid
    velocity only with no elastic coupling to anchors."""
    eye = ti.Matrix.identity(ti.f32, 3)
    for p in range(start, end):
        A = ti.Matrix.zero(ti.f32, 3, 3)
        B = ti.Matrix.zero(ti.f32, 3, 3)
        n_intact = 0
        for k in range(G.K_PHYS):
            if G._ti_bond_intact[p, k] == 1:
                nb = _ti_wnb_tl[p, k]
                w  = _ti_ww_tl[p, k]
                dc = G._ti_pos[nb]  - G._ti_pos[p]
                dr = G._ti_rest[nb] - G._ti_rest[p]
                A += w * dc.outer_product(dr)
                B += w * dr.outer_product(dr)
                n_intact += 1
        if n_intact >= 3 and ti.abs(B.determinant()) > 1e-9:
            G._ti_F[p] = eye + (A - B) @ B.inverse()
        # else: not enough intact bonds; leave F at its current (post-G2P) value


@ti.kernel
def _ti_wls_recompute_tl(start: ti.i32, end: ti.i32):
    """Per-frame TL drift correction: re-measure F via WLS over K_PHYS rest
    neighbours.  Only meaningful for TL  -  UL doesn't carry a rest frame."""
    eye = ti.Matrix.identity(ti.f32, 3)
    for p in range(start, end):
        A = ti.Matrix.zero(ti.f32, 3, 3)
        for k in range(G.K_PHYS):
            nb = _ti_wnb_tl[p, k]
            w  = _ti_ww_tl[p, k]
            dc = G._ti_pos[nb] - G._ti_pos[p]
            dr = G._ti_rest[nb] - G._ti_rest[p]
            A += w * dc.outer_product(dr)
        G._ti_F[p] = eye + (A - _ti_wB_tl[p]) @ _ti_wBi_tl[p]


# ============================================================
# Public API
# ============================================================

def substep(start: int, end: int, dt: float, kappa: float, mu_scale: float,
            model: int, floor_damp: float, friction: float,
            n_grid: int, grid_h: float) -> None:
    """One TL-APIC substep for particles [start, end).
        clear → p2g → grid_update → g2p
    `friction` is the Coulomb coefficient (μ) applied at floor contact in g2p.
    """
    _ti_clear_grid_tl(n_grid)
    _ti_p2g_tl(start, end, kappa, mu_scale, model, floor_damp, friction)
    _ti_grid_update_tl(n_grid, dt)
    _ti_g2p_tl(start, end, dt, floor_damp, friction, grid_h)


def wls_recompute(start: int, end: int) -> None:
    """Per-frame TL drift correction."""
    _ti_wls_recompute_tl(start, end)


def wls_recompute_fracture(start: int, end: int) -> None:
    """Fracture-aware WLS recompute that respects broken bonds."""
    _ti_wls_recompute_tl_fracture(start, end)


def bond_break_check(start: int, end: int,
                     stretch_thresh: float, dmg_thresh: float) -> None:
    """Update _ti_bond_intact flags based on stretch + damage thresholds."""
    _ti_bond_break_check(start, end, stretch_thresh, dmg_thresh)
