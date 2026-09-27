"""Run a GaussianBench scene in PhysGaussian (Warp MPM) and export a
submission.

Usage (from the PhysGaussian repo root, using its venv):
    .\\.venv\\Scripts\\python.exe -m bench.run_bench_scene
        --scene <path\\to\\scene.json> --out <dir> [--tier-c]
        [--max-seconds S] [--substep-dt DT]

Integration notes (from inspecting mpm_solver_warp/mpm_solver_warp.py):
  - MPM_Simulator_WARP: UL MLS-MPM on a dense [0, grid_lim]^3 grid,
    n_grid nodes per side, Warp f32 on CUDA.
  - load_initial_data_from_torch(x, vol): frozen scene positions go in
    verbatim (translated by a constant placement offset so the body sits
    inside the corner-anchored domain; exact inverse applied on export).
  - Material: "jelly" (fixed corotated) with the scene's E, nu, density.
    E and nu are passed straight through; the solver derives mu and
    lambda itself (finalize_mu_lam), so no conversion happens here at all.
  - p2g2p(step, dt) is one substep; self.time advances internally and
    drives the time windows of every boundary helper.
  - Boundary conditions use PhysGaussian's own primitives at BOTH levels,
    mirroring the GaussianFlesh entrant's finding that particle-level
    constraints alone are mass-diluted by the B-spline stencil:
      * grid level: set_velocity_on_cuboid (Dirichlet velocity boxes;
        a box with nonzero velocity advects itself, which is exactly a
        moving-face ramp)
      * particle level: per-substep velocity assignment through
        import_particle_v_from_torch for masked particles (equivalent to
        the solver's own particle_velocity_modifiers, but allows
        time-varying and spatially-varying prescriptions such as the
        half-sine pulse and the affine shell)
  - APIC affine matrix seeded via import_particle_C_from_torch with
    skew(omega) for rigid-spin initial conditions (C = 0 loses
    (dx/R)^2 of angular momentum on the first transfer).
  - Tier C covariances: PhysGaussian's own rendering pipeline computes
    deformed covariances as F Sigma_0 F^T from the deformation gradient;
    the harness reproduces exactly that from export_particle_F_to_torch
    with isotropic rest covariances at half the particle spacing.
  - No thermal model exists in PhysGaussian: the melt scenes are not
    submitted and score NA, which is the benchmark's intended semantics
    for a missing capability.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(os.environ.get(
    "GAUSSIANBENCH_METHOD_ROOT", Path(__file__).parent.parent)).resolve()
SYSTEM_NAME = os.environ.get("GAUSSIANBENCH_SYSTEM_NAME", "PhysGaussian")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

import warp as wp  # noqa: E402

from benchexport import BenchRecorder  # noqa: E402

DEVICE = "cuda:0"


# ---------------------------------------------------------------------------
# Planning (mirrors the GaussianFlesh harness where the logic is shared)
# ---------------------------------------------------------------------------

def load_scene(scene_path: Path):
    with open(scene_path, "r", encoding="utf-8") as f:
        scene = json.load(f)
    npy = scene_path.parent / scene["geometry"]["particle_positions_file"]
    return scene, np.load(npy).astype(np.float64)


def parse_region(expr: str, rest: np.ndarray) -> np.ndarray:
    parts = expr.split()
    axis = {"x": 0, "y": 1, "z": 2}[parts[0]]
    val = float(parts[2])
    col = rest[:, axis]
    return col < val if parts[1] == "<" else col > val


def plan_run(scene, scene_pos, args):
    if "thermal" in scene.get("material", {}):
        raise SystemExit(
            f"NOT SUBMITTED: {SYSTEM_NAME} has no thermal model (no heat "
            "transport, no phase change), so the melt scenes cannot be "
            "run. The benchmark marks them NA, which is the honest "
            "outcome for a missing capability.")
    gravity = np.asarray(scene["simulation"]["gravity"], float)

    fps = float(scene["simulation"]["output_fps"])
    duration = float(scene["simulation"]["duration_s"])
    if args.max_seconds is not None:
        duration = min(duration, float(args.max_seconds))
    n_frames = int(round(duration * fps)) + 1
    frame_dt = 1.0 / fps

    spacing = float(scene["geometry"]["particle_spacing_m"])
    grid_dx_target = 2.0 * spacing

    bbox_lo = scene_pos.min(axis=0)
    bbox_hi = scene_pos.max(axis=0)
    body_extent = float((bbox_hi - bbox_lo).max())
    ic = scene.get("initial_conditions", {})
    drift = np.zeros(3)
    if "linear_velocity_m_s" in ic:
        drift = np.asarray(ic["linear_velocity_m_s"], float) * duration
    margin = 4.0 * grid_dx_target + 0.1 * float(
        np.linalg.norm(bbox_hi - bbox_lo))
    co_moving = (not scene.get("boundary_conditions")
                 and float(np.linalg.norm(drift)) > body_extent)
    if co_moving:
        lo, hi = bbox_lo - margin, bbox_hi + margin
    else:
        lo = np.minimum(bbox_lo, bbox_lo + drift) - margin
        hi = np.maximum(bbox_hi, bbox_hi + drift) + margin
    grid_lim = float((hi - lo).max())
    n_grid = int(np.ceil(grid_lim / grid_dx_target))
    grid_dx = grid_lim / n_grid

    # Corner-anchored domain [0, grid_lim]^3: center the motion envelope.
    center = 0.5 * (lo + hi)
    offset = 0.5 * grid_lim - center

    E = float(scene["material"]["youngs_modulus_pa"])
    nu = float(scene["material"]["poisson_ratio"])
    density = float(scene["material"]["density_kg_m3"])
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c_wave = float(np.sqrt((lam + 2.0 * mu) / density))
    dt_cfl = 0.3 * grid_dx / max(c_wave, 1e-9)
    dt_sub = float(args.substep_dt) if args.substep_dt else min(dt_cfl,
                                                                frame_dt)
    substeps = max(1, int(np.ceil(frame_dt / dt_sub)))
    dt_sub = frame_dt / substeps

    return {
        "gravity": gravity, "fps": fps, "duration": duration,
        "n_frames": n_frames, "frame_dt": frame_dt, "dt_sub": dt_sub,
        "substeps": substeps, "grid_dx": grid_dx, "grid_lim": grid_lim,
        "n_grid": n_grid, "offset": offset, "co_moving": co_moving,
        "E": E, "nu": nu, "density": density,
        "mass": density * spacing ** 3, "spacing": spacing,
        "n_particles": len(scene_pos),
    }


def initial_velocities(scene, scene_pos):
    ic = scene.get("initial_conditions", {})
    n = len(scene_pos)
    vel = np.zeros((n, 3))
    if "linear_velocity_m_s" in ic:
        vel += np.asarray(ic["linear_velocity_m_s"], float)[None, :]
    if "angular_velocity_rad_s" in ic:
        omega = np.asarray(ic["angular_velocity_rad_s"], float)
        x_cm = scene_pos.mean(axis=0)
        vel += np.cross(omega[None, :], scene_pos - x_cm[None, :])
    if ic.get("type") == "modal_velocity_pluck":
        b = 1.875104068711961
        sigma = (np.cosh(b) - np.cos(b)) / (np.sinh(b) + np.sin(b))
        x = scene_pos[:, 0]
        s = (x - x.min()) / max(x.max() - x.min(), 1e-12)
        phi = (np.cosh(b * s) - np.cos(b * s)
               - sigma * (np.sinh(b * s) - np.sin(b * s)))
        phi_tip = (np.cosh(b) - np.cos(b) - sigma * (np.sinh(b) - np.sin(b)))
        direction = np.asarray(ic["direction"], float)
        vel += (float(ic["tip_velocity_m_s"]) * (phi / phi_tip))[:, None] \
            * direction[None, :]
    return vel


# ---------------------------------------------------------------------------
# Per-substep particle velocity prescriptions (masked, time varying)
# ---------------------------------------------------------------------------

def register_affine_grid_bc(solver, A, about_sim, T, shell_lo0, shell_hi0,
                            outer_pad):
    """Inject an exact affine Dirichlet velocity field over the moving
    boundary shell into the solver's grid_postprocess pipeline.

    A node at current position x is pulled back to rest coordinates
    x0 = M(t)^-1 (x - about) + about, M(t) = I + (t/T) A. If x0 lies in
    the shell (outside the inner rest box but within outer_pad of the
    body), the node velocity is set to the prescribed (1/T) A (x0 - about),
    which is constant per material point, so the field follows the
    deforming faces exactly. After t = T the shell is held at rest.
    """
    from mpm_solver_warp.mpm_solver_warp import (Dirichlet_collider,
                                                 MPMModelStruct,
                                                 MPMStateStruct)

    A = np.asarray(A, float)
    assert np.allclose(A, np.diag(np.diag(A))), \
        "affine grid BC supports diagonal F_target only (all it needs)"

    # No closures: every value rides in the param struct, matching the
    # solver's own collider pattern (Warp kernels defined in methods there
    # reference only their arguments). Field reuse, documented:
    #   point    -> about (sim coords)      normal -> inner rest box lo
    #   size     -> inner rest box hi       velocity -> diag(A)
    #   friction -> outer pad               start/end -> 0 / ramp end T
    param = Dirichlet_collider()
    param.point = wp.vec3(*[float(v) for v in about_sim])
    param.normal = wp.vec3(*[float(v) for v in shell_lo0])
    param.size = wp.vec3(*[float(v) for v in shell_hi0])
    param.velocity = wp.vec3(float(A[0, 0]), float(A[1, 1]), float(A[2, 2]))
    param.friction = float(outer_pad)
    param.start_time = 0.0
    param.end_time = float(T)

    @wp.kernel
    def affine_shell_bc(time: float, dt: float, state: MPMStateStruct,
                        model: MPMModelStruct, param: Dirichlet_collider):
        gx, gy, gz = wp.tid()
        x = wp.vec3(float(gx) * model.dx, float(gy) * model.dx,
                    float(gz) * model.dx)
        T_end = param.end_time
        alpha = wp.min(time / T_end, 1.0)
        # rest pull-back, diagonal M(t) = I + alpha diag(A)
        x0 = wp.vec3(
            (x[0] - param.point[0]) / (1.0 + alpha * param.velocity[0])
            + param.point[0],
            (x[1] - param.point[1]) / (1.0 + alpha * param.velocity[1])
            + param.point[1],
            (x[2] - param.point[2]) / (1.0 + alpha * param.velocity[2])
            + param.point[2])
        pad = param.friction
        inside_inner = (x0[0] > param.normal[0] and x0[0] < param.size[0]
                        and x0[1] > param.normal[1]
                        and x0[1] < param.size[1]
                        and x0[2] > param.normal[2]
                        and x0[2] < param.size[2])
        near_body = (x0[0] > param.normal[0] - pad
                     and x0[0] < param.size[0] + pad
                     and x0[1] > param.normal[1] - pad
                     and x0[1] < param.size[1] + pad
                     and x0[2] > param.normal[2] - pad
                     and x0[2] < param.size[2] + pad)
        if near_body and not inside_inner:
            v = wp.vec3(0.0, 0.0, 0.0)
            if time < T_end:
                v = wp.vec3(
                    param.velocity[0] * (x0[0] - param.point[0]) / T_end,
                    param.velocity[1] * (x0[1] - param.point[1]) / T_end,
                    param.velocity[2] * (x0[2] - param.point[2]) / T_end)
            state.grid_v_out[gx, gy, gz] = v

    solver.collider_params.append(param)
    solver.grid_postprocess.append(affine_shell_bc)
    solver.modify_bc.append(None)


def register_squash_plate_grid_bc(solver, floor_y, top_y, drop, press_end,
                                  hold_end):
    """Register the C3 unilateral floor and moving top plate on the grid."""
    from mpm_solver_warp.mpm_solver_warp import (Dirichlet_collider,
                                                 MPMModelStruct,
                                                 MPMStateStruct)

    param = Dirichlet_collider()
    # point = [floor_y, top_y, drop], normal = [press_end, hold_end, 0].
    param.point = wp.vec3(float(floor_y), float(top_y), float(drop))
    param.normal = wp.vec3(float(press_end), float(hold_end), 0.0)
    param.start_time = 0.0
    param.end_time = float(hold_end)

    @wp.kernel
    def squash_plate_bc(time: float, dt: float, state: MPMStateStruct,
                        model: MPMModelStruct, param: Dirichlet_collider):
        gx, gy, gz = wp.tid()
        y = float(gy) * model.dx
        velocity = state.grid_v_out[gx, gy, gz]

        if y < param.point[0] + 0.75 * model.dx and velocity[1] < 0.0:
            velocity = wp.vec3(velocity[0], 0.0, velocity[2])

        if time <= param.normal[1]:
            alpha = wp.min(time / param.normal[0], 1.0)
            plane = param.point[1] - alpha * param.point[2]
            plate_velocity = 0.0
            if time < param.normal[0]:
                plate_velocity = -param.point[2] / param.normal[0]
            if y > plane - 0.75 * model.dx and velocity[1] > plate_velocity:
                velocity = wp.vec3(
                    velocity[0], plate_velocity, velocity[2])

        state.grid_v_out[gx, gy, gz] = velocity

    solver.collider_params.append(param)
    solver.grid_postprocess.append(squash_plate_bc)
    solver.modify_bc.append(None)


def register_simple_shear_grid_bc(solver, lo0, hi0, gamma_peak, duration,
                                  face_band, outer_pad):
    """Drive the y-min/y-max faces with the scene's triangular xy shear.

    The pull-back x0 = x - gamma(t) y e_x makes face selection follow the
    moving material.  This uses the same grid_postprocess hook as the
    production colliders, so the prescribed face motion is transmitted
    through P2G rather than being diluted by a particle-only overwrite.
    """
    from mpm_solver_warp.mpm_solver_warp import (Dirichlet_collider,
                                                 MPMModelStruct,
                                                 MPMStateStruct)

    param = Dirichlet_collider()
    param.point = wp.vec3(*[float(v) for v in lo0])
    param.size = wp.vec3(*[float(v) for v in hi0])
    param.normal = wp.vec3(float(gamma_peak), float(duration),
                           float(face_band))
    param.friction = float(outer_pad)
    param.start_time = 0.0
    param.end_time = float(duration)

    @wp.kernel
    def simple_shear_face_bc(time: float, dt: float, state: MPMStateStruct,
                             model: MPMModelStruct,
                             param: Dirichlet_collider):
        gx, gy, gz = wp.tid()
        x = wp.vec3(float(gx) * model.dx, float(gy) * model.dx,
                    float(gz) * model.dx)
        T = param.normal[1]
        half = 0.5 * T
        gamma = 0.0
        rate = 0.0
        if time < half:
            gamma = param.normal[0] * time / half
            rate = param.normal[0] / half
        elif time < T:
            gamma = param.normal[0] * (T - time) / half
            rate = -param.normal[0] / half

        # The benchmark shear is about scene y=0.  In the translated solver
        # frame that origin is param.point.y, so both pull-back and velocity
        # must use y_rel rather than the absolute grid coordinate.
        y_rel = x[1] - param.point[1]
        x0 = wp.vec3(x[0] - gamma * y_rel, x[1], x[2])
        pad = param.friction
        near_body = (x0[0] > param.point[0] - pad
                     and x0[0] < param.size[0] + pad
                     and x0[2] > param.point[2] - pad
                     and x0[2] < param.size[2] + pad)
        on_face = (x0[1] < param.point[1] + param.normal[2]
                   or x0[1] > param.size[1] - param.normal[2])
        if near_body and on_face:
            state.grid_v_out[gx, gy, gz] = wp.vec3(rate * y_rel, 0.0, 0.0)

    solver.collider_params.append(param)
    solver.grid_postprocess.append(simple_shear_face_bc)
    solver.modify_bc.append(None)


class ParticlePrescription:
    """mask (N,) bool; vel_fn(t) -> (N, 3) target velocities for masked
    particles (full array, only masked rows used); optional pos_fn(t) ->
    (N, 3) exact target positions (velocity-only enforcement lets pinned
    particles lag their prescription, since positions integrate through
    the mass-diluted grid field); active_fn(t)."""

    def __init__(self, mask, vel_fn, active_fn=None, pos_fn=None):
        self.mask = torch.from_numpy(mask.copy()).to(DEVICE)
        self.vel_fn = vel_fn
        self.pos_fn = pos_fn
        self.active_fn = active_fn or (lambda t: True)


def build_bcs(scene, scene_pos, offset, solver, plan, notes):
    """Map scene BCs onto PhysGaussian primitives. Returns particle
    prescriptions applied per substep; grid-level Dirichlet boxes are
    registered directly on the solver."""
    prescriptions = []
    bcs = scene.get("boundary_conditions", {})
    spacing = plan["spacing"]
    pad = 0.8 * plan["grid_dx"]

    def box_of(mask, extra=pad):
        pts = scene_pos[mask] + offset[None, :]
        center = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
        half = 0.5 * (pts.max(axis=0) - pts.min(axis=0)) + extra
        return center, half

    for name, bc in bcs.items():
        if not isinstance(bc, dict):
            continue
        btype = bc.get("type", "")
        if btype == "fixed":
            mask = parse_region(bc["region"], scene_pos)
            c, h = box_of(mask)
            solver.set_velocity_on_cuboid(c, list(h), [0.0, 0.0, 0.0],
                                          0.0, 1e9)
            zero = np.zeros((len(scene_pos), 3))
            prescriptions.append(ParticlePrescription(
                mask, lambda t, z=zero: z))
        elif btype == "displacement_ramp":
            mask = parse_region(bc["region"], scene_pos)
            dvec = np.asarray(bc["displacement_m"], float)
            T = float(bc["ramp_end_s"])
            v_ramp = dvec / T
            c, h = box_of(mask)
            # moving box during the ramp (advects itself at its velocity),
            # then a static hold box at the final position
            solver.set_velocity_on_cuboid(c, list(h), list(v_ramp), 0.0, T)
            solver.set_velocity_on_cuboid(list(np.asarray(c) + dvec),
                                          list(h), [0.0, 0.0, 0.0], T, 1e9)
            n = len(scene_pos)

            def vfn(t, n=n, v=v_ramp, T=T):
                out = np.zeros((n, 3))
                if t < T:
                    out[:] = v
                return out

            prescriptions.append(ParticlePrescription(mask, vfn))
        elif btype == "prescribed_displacement_pulse":
            mask = parse_region(bc["region"], scene_pos)
            avec = (float(bc["amplitude_m"])
                    * np.asarray(bc["direction"], float))
            W = float(bc["width_s"])
            c, h = box_of(mask, extra=pad + float(np.abs(avec).max()))
            # one grid box whose velocity the harness rewrites per substep
            solver.set_velocity_on_cuboid(c, list(h), [0.0, 0.0, 0.0],
                                          0.0, 1e9)
            pulse_param = solver.collider_params[-1]
            n = len(scene_pos)

            def vfn(t, n=n, a=avec, W=W, param=pulse_param):
                rate = (np.pi / W) * np.cos(np.pi * t / W) if t <= W else 0.0
                v = rate * a
                param.velocity = wp.vec3(float(v[0]), float(v[1]),
                                         float(v[2]))
                out = np.zeros((n, 3))
                out[:] = v
                return out

            prescriptions.append(ParticlePrescription(mask, vfn))
            notes.append(
                "Pulse face: grid Dirichlet box with per-substep velocity "
                "rewrite following the scene's half-sine formula exactly; "
                "face held at rest after the pulse window.")
        elif btype == "displacement_ramp_affine":
            F_target = np.asarray(bc["F_target"], float)
            about = np.asarray(bc.get("about", [0, 0, 0]), float)
            lo = scene_pos.min(axis=0) + spacing * 1.6
            hi = scene_pos.max(axis=0) - spacing * 1.6
            mask = ~np.all((scene_pos > lo) & (scene_pos < hi), axis=1)
            A = F_target - np.eye(3)
            T = float(bc["ramp_end_s"])
            x0_rel = scene_pos - about

            def vfn(t, A=A, T=T, x0=x0_rel, n=len(scene_pos)):
                if t >= T:
                    return np.zeros((n, 3))
                # exact rest-coordinate prescription: v = (1/T) A x0
                return x0 @ (A.T / T)

            base_sim = scene_pos + offset[None, :]

            def pfn(t, A=A, T=T, x0=x0_rel, base=base_sim):
                a = min(t / T, 1.0)
                return base + a * (x0 @ A.T)

            prescriptions.append(ParticlePrescription(mask, vfn,
                                                      pos_fn=pfn))
            # Grid-level transmission: a custom Warp kernel injected into
            # the solver's grid_postprocess list (the same slot its own
            # colliders use). It prescribes the EXACT affine field
            # v(x) = (1/T) A M(t)^-1 (x - about),  M(t) = I + (t/T) A
            # on grid nodes whose rest pull-back lies in the boundary
            # shell. Uniform-velocity cuboids cannot express the
            # tangential variation of this field (measured: median
            # covariance error 0.31 with face boxes, 0.50 with
            # particle-only prescriptions, i.e. interior F = I).
            register_affine_grid_bc(
                solver,
                A=A, about_sim=about + offset, T=T,
                shell_lo0=lo + offset, shell_hi0=hi + offset,
                outer_pad=4.0 * plan["grid_dx"])
            notes.append(
                "Affine ramp: two-layer particle shell velocity-prescribed "
                "per substep in exact rest coordinates, plus a custom grid "
                "Dirichlet kernel (registered through the solver's own "
                "grid_postprocess mechanism) applying the exact affine "
                "velocity field over the moving boundary shell; "
                "particle-only prescriptions are mass-diluted by the "
                "transfer stencil and do not transmit.")
        elif btype == "prescribed_simple_shear":
            gamma = float(bc["gamma_peak"])
            y = scene_pos[:, 1]
            mask = (y < y.min() + spacing * 1.6) \
                | (y > y.max() - spacing * 1.6)
            T = float(scene["simulation"]["duration_s"])
            n = len(scene_pos)

            def vfn(t, g=gamma, T=T, y=y, n=n):
                rate = (2.0 / T) if t < T / 2 else (-2.0 / T)
                out = np.zeros((n, 3))
                out[:, 0] = rate * g * y
                return out

            base_sim = scene_pos + offset[None, :]

            def pfn(t, g=gamma, T=T, base=base_sim, y=y):
                tri = 2.0 * t / T if t < T / 2 else 2.0 * (1.0 - t / T)
                out = base.copy()
                out[:, 0] += g * max(tri, 0.0) * y
                return out

            prescriptions.append(ParticlePrescription(mask, vfn,
                                                      pos_fn=pfn))
            register_simple_shear_grid_bc(
                solver, scene_pos.min(axis=0) + offset,
                scene_pos.max(axis=0) + offset, gamma, T,
                face_band=1.6 * spacing, outer_pad=4.0 * plan["grid_dx"])
            notes.append(
                "Simple shear: y-min/y-max particle faces follow the exact "
                "triangular x += gamma(t)y prescription, with the matching "
                "moving-face velocity imposed through PhysGaussian's "
                "grid_postprocess collider hook.")
        elif btype == "displacement_controlled_plate":
            # unilateral moving plane + floor plane, handled in main loop
            continue
        elif btype == "fixed_temperature":
            continue
    return prescriptions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tier-c", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--substep-dt", type=float, default=None)
    args = ap.parse_args(argv)

    scene, scene_pos = load_scene(args.scene)
    plan = plan_run(scene, scene_pos, args)
    notes = []
    n = plan["n_particles"]

    print(f"[bench] {scene['scene_id']}: {n} particles, "
          f"{plan['n_frames']} frames at {plan['fps']:.0f} fps, "
          f"dt {plan['dt_sub']:.2e} x {plan['substeps']}/frame, "
          f"grid {plan['n_grid']}^3 dx={plan['grid_dx']:.4f}")

    wp.init()
    from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP

    solver = MPM_Simulator_WARP(10)  # re-initialized by the loader

    sim_pos = scene_pos + plan["offset"][None, :]
    x_t = torch.from_numpy(sim_pos.astype(np.float32)).to(DEVICE)
    vol_t = torch.full((n,), plan["spacing"] ** 3,
                       dtype=torch.float32).to(DEVICE)
    sigma0 = (plan["spacing"] * 0.5) ** 2
    rcov = np.tile((np.eye(3, dtype=np.float32) * sigma0)[None],
                   (n, 1, 1))
    # PhysGaussian's native simulation path stores each symmetric rest
    # covariance as [xx, xy, xz, yy, yz, zz].  Supplying it here lets the
    # benchmark read covariance through the exact production export path
    # instead of reconstructing a parallel adapter-side implementation.
    rcov_upper = np.stack(
        [rcov[:, 0, 0], rcov[:, 0, 1], rcov[:, 0, 2],
         rcov[:, 1, 1], rcov[:, 1, 2], rcov[:, 2, 2]], axis=1)
    solver.load_initial_data_from_torch(
        x_t, vol_t, torch.from_numpy(rcov_upper).to(DEVICE),
        n_grid=plan["n_grid"], grid_lim=plan["grid_lim"], device=DEVICE)
    native_params = {
        "material": "jelly_base" if SYSTEM_NAME == "Physics3D" else "jelly",
        "E": plan["E"], "nu": plan["nu"], "density": plan["density"],
        "g": list(plan["gravity"]),
        "grid_v_damping_scale": 1.1,  # >=1 disables the damping kernel
    }
    if SYSTEM_NAME == "Physics3D":
        # Released Physics3D's elastic branch adds a learned Maxwell term.
        # GaussianBench requests the declared corotated law, so its residual
        # moduli are zero; viscosity must still be positive because leaving
        # it unset produces an artificial 0/0 NaN in the released kernel.
        native_params.update({"mu_N": 0.0, "lam_N": 0.0,
                              "viscosity": 10.0})
        # Physics3D stores Young's modulus in units of 1e7 Pa: its released
        # compute_mu_lam_from_E_nu multiplies E by 1e7 (native ball config
        # E=0.2 for 2 MPa). Passing raw Pa makes mu 1e7x too large, violates
        # CFL by ~3000x, and crashes with an out-of-bounds p2g atomic_add
        # (CUDA illegal memory access). Substep planning above already uses
        # the physical wave speed, which stays correct under this mapping.
        native_params["E"] = plan["E"] / 1.0e7
    solver.set_parameters_dict(native_params, device=DEVICE)
    solver.finalize_mu_lam(device=DEVICE)
    passed_how = ("passed as E/1e7 in the solver's native E units (see "
                  "parameter-mapping note)" if SYSTEM_NAME == "Physics3D"
                  else "passed straight through")
    notes.append(
        f"Solver: {SYSTEM_NAME} native UL MLS-MPM (Warp, cuda, f32), material "
        f"'{native_params['material']}' (fixed corotated) with scene "
        f"E={plan['E']:g}, nu={plan['nu']:g}, density={plan['density']:g} "
        f"{passed_how} (the solver derives Lame parameters itself). "
        f"frame_dt={plan['frame_dt']:.6g}, substep_dt={plan['dt_sub']:.6g}, "
        f"{plan['substeps']} substeps/frame, grid {plan['n_grid']}^3, "
        "grid damping disabled.")
    if SYSTEM_NAME == "Physics3D":
        notes.append(
            "Parameter mapping: scene E [Pa] passed to the solver as E/1e7 "
            "because Physics3D's released compute_mu_lam_from_E_nu stores E "
            "in units of 1e7 Pa (native ball config E=0.2); the resulting "
            "Lame parameters equal the scene's physical values exactly.")
        notes.append(
            "Reproduction fix: initialized released particle_F_N persistent "
            "Maxwell state to identity, matching particle_F_N_trial. The "
            "released loader left particle_F_N at zero, causing SVD/log of "
            "a singular state on the first constitutive update. Maxwell "
            "residual moduli are zero for the benchmark's declared pure "
            "corotated material.")

    # Initial conditions
    vel0 = initial_velocities(scene, scene_pos)
    boost = np.zeros(3)
    if plan["co_moving"]:
        boost = np.asarray(
            scene["initial_conditions"]["linear_velocity_m_s"], float)
        vel0 = vel0 - boost[None, :]
    solver.import_particle_v_from_torch(
        torch.from_numpy(vel0.astype(np.float32)).to(DEVICE), device=DEVICE)
    ic = scene.get("initial_conditions", {})
    if "angular_velocity_rad_s" in ic:
        wx, wy, wz = np.asarray(ic["angular_velocity_rad_s"], float)
        skew = np.array([[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]],
                        dtype=np.float32)
        C0 = np.tile(skew, (n, 1, 1))
        solver.import_particle_C_from_torch(
            torch.from_numpy(C0).to(DEVICE), device=DEVICE)
        notes.append("APIC affine matrix seeded with skew(omega) for the "
                     "rigid-spin initial condition (C = 0 filters "
                     "(dx/R)^2 of L on the first transfer).")

    prescriptions = build_bcs(scene, scene_pos, plan["offset"], solver,
                              plan, notes)

    # C3 unilateral top plate and floor. Grid constraints transmit the
    # boundary into the MPM body; a particle clamp after every substep
    # prevents finite-stencil leakage through either plane.
    plate = None
    plate_bc = scene.get("boundary_conditions", {}).get("squash_plate")
    if (isinstance(plate_bc, dict)
            and plate_bc.get("type") == "displacement_controlled_plate"):
        y_sim = sim_pos[:, 1]
        initial_height = float(y_sim.max() - y_sim.min())
        drop = (
            1.0 - float(plate_bc["target_height_frac"])
        ) * initial_height
        plate = {
            "floor": float(y_sim.min()),
            "top": float(y_sim.max()),
            "drop": drop,
            "press": float(plate_bc["press_end_s"]),
            "hold": float(plate_bc["hold_end_s"]),
        }
        register_squash_plate_grid_bc(
            solver, plate["floor"], plate["top"], plate["drop"],
            plate["press"], plate["hold"])
        notes.append(
            "C3 boundary: unilateral floor and descending top plate applied "
            "to PhysGaussian's grid after grid update and before G2P, with "
            "a particle-level no-penetration clamp after each substep. The "
            "plate reaches 80% rest height at 1.0 s, holds to 1.5 s, then "
            "is removed exactly as specified by the scene.")

    if plan["co_moving"]:
        notes.append(
            "Co-moving frame (Galilean boost): simulated in the frame "
            f"translating at v_frame={boost.tolist()} m/s; exact inverse "
            "(x + v_frame t, v + v_frame) applied on export.")

    rec = BenchRecorder(
        args.out / scene["scene_id"], args.scene,
        record_velocities=True, record_mass=True,
        record_covariances=bool(args.tier_c),
        record_deformation_gradients=bool(args.tier_c),
        scene_to_sim={"offset": plan["offset"].tolist(), "permutation": None,
                      "frame_velocity":
                          boost.tolist() if plan["co_moving"] else None},
        notes=notes)
    if args.max_seconds is not None:
        rec.notes.append(
            f"DEVIATION: duration truncated to {plan['duration']:g} s by "
            "--max-seconds (dry run, not a scoreable submission).")
    req = scene.get("pass_criteria", {}).get("requires_meta_declaration", {})
    extra_meta = {"scene_id": scene["scene_id"]}
    if req:
        strategy = "co_moving" if plan["co_moving"] else "fixed"
        extra_meta["notes"] = {"domain_handling": strategy,
                               "text": " | ".join(rec.notes)}

    def state(first=False):
        st = {
            "positions": solver.export_particle_x_to_torch().detach().cpu().numpy(),
            "velocities": solver.export_particle_v_to_torch().detach().cpu().numpy(),
            "mass": np.full(n, plan["mass"], dtype=np.float32),
        }
        if args.tier_c:
            if first:
                st["deformation_gradients"] = np.tile(
                    np.eye(3, dtype=np.float32), (n, 1, 1))
            else:
                st["deformation_gradients"] = (
                    solver.export_particle_F_to_torch().detach().cpu().numpy())
            if first:
                # Warp's particle_F is zeros until the first p2g2p (only
                # F_trial is initialized to identity); F = I at t = 0 by
                # definition of the initial condition, so frame 0
                # covariances are the rest covariances.
                st["covariances"] = rcov.copy()
            else:
                # This is the same covariance export called by
                # gs_simulation.py before rasterization.  The solver's
                # compute_cov_from_F kernel performs F Sigma_0 F^T.
                upper = (solver.export_particle_cov_to_torch()
                         .reshape(n, 6).detach().cpu().numpy())
                cov = np.empty((n, 3, 3), dtype=np.float32)
                cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2] = \
                    upper[:, 0], upper[:, 1], upper[:, 2]
                cov[:, 1, 0], cov[:, 1, 1], cov[:, 1, 2] = \
                    upper[:, 1], upper[:, 3], upper[:, 4]
                cov[:, 2, 0], cov[:, 2, 1], cov[:, 2, 2] = \
                    upper[:, 2], upper[:, 4], upper[:, 5]
                st["covariances"] = cov
        return st

    c3_renderer = None
    c3_camera = None
    if scene["scene_id"].startswith("c3_"):
        from physgaussian_renderer import PhysGaussianRenderer

        rgb = 0.5 + 0.35 * np.sin(
            scene_pos @ np.array(
                [[41.0, 13.0, 7.0],
                 [11.0, 47.0, 17.0],
                 [5.0, 19.0, 53.0]]).T)
        sh = np.zeros((n, 16, 3), dtype=np.float32)
        sh[:, 0, :] = ((rgb - 0.5) / 0.28209479).astype(np.float32)
        opacity = np.full((n, 1), 0.85, dtype=np.float32)
        c3_renderer = PhysGaussianRenderer(opacity, sh)
        center = sim_pos.mean(axis=0).astype(np.float64)
        radius = float(np.linalg.norm(sim_pos - center, axis=1).max())
        eye = center + np.array([0.35, 0.3, 0.9]) * (3.2 * radius)
        c3_camera = {
            "eye": eye,
            "target": center,
            "up": np.array([0.0, 1.0, 0.0]),
        }
        rec.write_camera({
            "position": (eye - plan["offset"]).tolist(),
            "look_at": (center - plan["offset"]).tolist(),
            "up": c3_camera["up"].tolist(),
            "fov_deg": 45.0,
            "width": 512,
            "height": 512,
            "note": (
                "Scene coordinates; rendered with PhysGaussian's vendored "
                "diff-gaussian-rasterization backend using its live "
                "positions and F Sigma_0 F^T covariances."
            ),
        })
        before = c3_renderer.render(
            sim_pos.astype(np.float32), rcov,
            c3_camera["eye"], c3_camera["target"], c3_camera["up"])
        rec.add_render(0, before, name="before.png")

    def apply_prescriptions(t):
        active = [p for p in prescriptions if p.active_fn(t)]
        if not active:
            return
        v = solver.export_particle_v_to_torch()
        for p in active:
            tgt = torch.from_numpy(
                p.vel_fn(t).astype(np.float32)).to(DEVICE)
            v = torch.where(p.mask[:, None], tgt, v)
        solver.import_particle_v_from_torch(v, device=DEVICE)
        if any(p.pos_fn is not None for p in active):
            x = solver.export_particle_x_to_torch()
            for p in active:
                if p.pos_fn is not None:
                    tgt = torch.from_numpy(
                        p.pos_fn(t).astype(np.float32)).to(DEVICE)
                    x = torch.where(p.mask[:, None], tgt, x)
            solver.import_particle_x_from_torch(x, device=DEVICE)

    def apply_plate_particles(t):
        if plate is None:
            return
        x = solver.export_particle_x_to_torch()
        v = solver.export_particle_v_to_torch()
        floor = torch.tensor(plate["floor"], device=DEVICE)
        below = x[:, 1] < floor
        x[:, 1] = torch.where(below, floor, x[:, 1])
        v[:, 1] = torch.where(
            below & (v[:, 1] < 0.0), torch.zeros_like(v[:, 1]), v[:, 1])

        if t <= plate["hold"]:
            alpha = min(t / plate["press"], 1.0)
            plane_value = plate["top"] - alpha * plate["drop"]
            plate_velocity = (
                -plate["drop"] / plate["press"]
                if t < plate["press"] else 0.0)
            plane = torch.tensor(plane_value, device=DEVICE)
            velocity = torch.tensor(plate_velocity, device=DEVICE)
            above = x[:, 1] > plane
            x[:, 1] = torch.where(above, plane, x[:, 1])
            v[:, 1] = torch.where(
                above & (v[:, 1] > velocity), velocity, v[:, 1])

        solver.import_particle_x_from_torch(x, device=DEVICE)
        solver.import_particle_v_from_torch(v, device=DEVICE)

    t_report = max(1, plan["n_frames"] // 10)
    rec.on_frame(state(first=True), 0, 0.0)
    step = 0
    for frame in range(1, plan["n_frames"]):
        for s in range(plan["substeps"]):
            apply_prescriptions(solver.time)
            solver.p2g2p(step, plan["dt_sub"], device=DEVICE)
            apply_plate_particles(solver.time)
            step += 1
        rec.on_frame(state(), frame, frame * plan["frame_dt"])
        if frame % t_report == 0:
            print(f"[bench] frame {frame}/{plan['n_frames'] - 1}")

    if c3_renderer is not None:
        final_state = state()
        after = c3_renderer.render(
            final_state["positions"], final_state["covariances"],
            c3_camera["eye"], c3_camera["target"], c3_camera["up"])
        rec.add_render(
            plan["n_frames"] - 1, after, name="after.png")

    out = rec.finalize(extra_meta=extra_meta)
    print(f"[bench] submission written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
