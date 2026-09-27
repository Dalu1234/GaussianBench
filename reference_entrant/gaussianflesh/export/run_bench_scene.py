"""Run a GaussianBench scene in GaussianFlesh and export a submission.

Usage (from the Representation repo root):
    python -m export.run_bench_scene --scene <path\\to\\scene.json> --out <dir>
        [--tier-c] [--max-seconds S] [--substep-dt DT]

Design constraints implemented here (see GaussianBench SPEC.md):
  - Scene positions are loaded VERBATIM from the scene's .npy. The only
    geometric liberty is a rigid placement transform (constant offset, plus
    a fixed axis permutation for z-up thermal scenes) so the body sits
    inside the UL world grid; the exact inverse is applied on export and
    the transform is recorded in meta.json notes.
  - Material mapping is the ONE permitted conversion:
        mu = E / (2 (1 + nu)),  lambda = E nu / ((1 + nu)(1 - 2 nu))
    Inputs and outputs go into meta.json notes.
  - No simulator source file is modified. The harness drives
    solver_ul.substep(..., use_floor=False) directly and calls the entity's
    _finish_frame() once per display frame (which is where GaussianFlesh
    itself updates covariances and syncs device to host). Boundary
    conditions the scenes need (fixed clamps, prescribed-displacement ramps)
    are applied by pin kernels defined in THIS module, operating on the
    simulator's fields between substeps.
  - Anything GaussianFlesh cannot express faithfully is refused loudly or
    declared in meta.json notes, never approximated silently.

Scene support:
  a1_conservation      full (the acceptance path)
  a2/c4 frequency      full (modal velocity pluck + clamp pin)
  a3 wavespeed         full (half-sine end pulse pin)
  a5 stiffness         full (fixed face + ramp face pins; lateral faces are
                       genuinely traction-free: the body floats far from the
                       domain walls and the floor BC is disabled)
  c1 covariance        full with --tier-c (all-faces affine ramp pin)
  c2 shear             full with --tier-c (top/bottom shear pins)
  c3 roundtrip         trajectory + covariances; RENDERS NOT WIRED yet
                       (gs_render appearance synthesis for procedural blobs
                       is a separate task), declared in notes
  a4 / c2 melt         REFUSED: GaussianFlesh's thermodynamics uses
                       iteration-calibrated diffusion, not SI conductivity,
                       so the scene's W/m/K cannot be mapped faithfully.
                       Submitting would misrepresent; skip and be marked NA.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent

_HARD_GRID_NODE_CAP = 150  # nodes per axis, mirrors solver_ul._HARD_GRID_CAP


# ---------------------------------------------------------------------------
# Scene analysis (pure numpy, before any Taichi import)
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger(__name__)

def load_scene(scene_path: Path) -> tuple[dict, np.ndarray]:
    with open(scene_path, "r", encoding="utf-8") as f:
        scene = json.load(f)
    npy = scene_path.parent / scene["geometry"]["particle_positions_file"]
    positions = np.load(npy).astype(np.float64)
    return scene, positions


def parse_region(expr: str, rest: np.ndarray) -> np.ndarray:
    """Evaluate a region predicate like 'x < 0.01' on scene rest positions."""
    parts = expr.split()
    if len(parts) != 3 or parts[0] not in ("x", "y", "z") \
            or parts[1] not in ("<", ">"):
        raise SystemExit(f"Cannot parse boundary region '{expr}'. "
                         "Supported form: '<axis> </> <value>'.")
    axis = {"x": 0, "y": 1, "z": 2}[parts[0]]
    val = float(parts[2])
    col = rest[:, axis]
    return col < val if parts[1] == "<" else col > val


def lame_from_scene(material: dict) -> tuple[float, float]:
    E = float(material["youngs_modulus_pa"])
    nu = float(material["poisson_ratio"])
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def plan_run(scene: dict, scene_pos: np.ndarray, args) -> dict:
    """Everything decided before the simulator is imported."""
    sim_cfg = scene["simulation"]
    gravity = np.asarray(sim_cfg["gravity"], dtype=np.float64)

    # Thermal (z-up) scenes run through a fixed axis permutation: scene z
    # (up, hot floor normal) maps to sim y. P = Rx(+90): (x,y,z) ->
    # (x, z, -y), proper rotation, exact inverse applied on export. The
    # heat solve itself uses solver_ul.heat_step_physical, which is
    # SI-exact (see its docstring); the original iteration-tuned heat_step
    # is not touched.
    thermal = scene.get("material", {}).get("thermal")
    permutation = None
    if thermal is not None:
        permutation = np.array([[1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0],
                                [0.0, -1.0, 0.0]])
        gravity = permutation @ gravity

    # Gravity: GaussianFlesh supports y-gravity only (after any thermal
    # permutation). Zero gravity is fine; anything else is unsupported.
    if np.allclose(gravity, 0.0):
        gravity_y = 0.0
    elif gravity[0] == 0.0 and gravity[2] == 0.0:
        gravity_y = float(gravity[1])
    else:
        raise SystemExit(f"REFUSED: gravity {gravity.tolist()} is not along "
                         "y and no supported axis mapping applies.")

    fps = float(sim_cfg["output_fps"])
    duration = float(sim_cfg["duration_s"])
    if args.max_seconds is not None:
        duration = min(duration, float(args.max_seconds))
    n_frames = int(round(duration * fps)) + 1
    frame_dt = 1.0 / fps

    spacing = float(scene["geometry"]["particle_spacing_m"])
    grid_dx = 2.0 * spacing

    # Motion envelope: bounding box of the body plus everywhere the initial
    # velocities can carry it (A1b drifts about 11.5 m; static scenes barely
    # move). Uniform-velocity drift is the only large-travel case.
    bbox_lo = scene_pos.min(axis=0)
    bbox_hi = scene_pos.max(axis=0)
    body_extent = float((bbox_hi - bbox_lo).max())
    ic = scene.get("initial_conditions", {})
    drift = np.zeros(3)
    if "linear_velocity_m_s" in ic:
        drift = np.asarray(ic["linear_velocity_m_s"], float) * duration
    margin = 4.0 * grid_dx + 0.1 * float(np.linalg.norm(bbox_hi - bbox_lo))

    # Co-moving domain: when the scene has no boundary conditions (nothing
    # anchored to world coordinates) and the drift dwarfs the body, size the
    # grid to the BODY and translate its origin with the body each frame
    # (solver_ul.set_grid_origin; exact, the grid holds no cross-substep
    # state). This keeps dx at 2x particle spacing instead of coarsening the
    # body into invisibility to cover the whole travel envelope.
    co_moving = (not scene.get("boundary_conditions")
                 and float(np.linalg.norm(drift)) > body_extent)
    if co_moving:
        lo = bbox_lo - margin
        hi = bbox_hi + margin
    else:
        lo = np.minimum(bbox_lo, bbox_lo + drift) - margin
        hi = np.maximum(bbox_hi, bbox_hi + drift) + margin
    extent = float((hi - lo).max())

    # If the required extent at 2x spacing exceeds the solver's hard node
    # cap, coarsen dx to fit (grid resolution is the submitter's freedom;
    # the frozen particles are the contract). Refuse only if even the
    # coarsest sane grid (4 cells across the body) cannot fit.
    n_nodes = int(np.ceil(extent / grid_dx)) + 1
    coarsened = False
    if n_nodes > _HARD_GRID_NODE_CAP:
        grid_dx = extent / (_HARD_GRID_NODE_CAP - 1) * 1.001
        coarsened = True
        body_cells = float((bbox_hi - bbox_lo).max()) / grid_dx
        if body_cells < 1.0:
            raise SystemExit(
                f"REFUSED: this scene needs a {extent:.1f} m domain, which "
                f"at the solver's {_HARD_GRID_NODE_CAP}^3 node cap forces "
                f"grid dx = {grid_dx:.3f} m, larger than the whole body. "
                "GaussianFlesh cannot allocate a grid this scene needs.")
    grid_lim = extent

    # Placement: grid domain is x,z in [-lim/2, lim/2], y in [0, lim].
    # Put the motion envelope centered in x,z and starting just inside y,
    # and always ABOVE the simulator's floor contact zone (y < 0.05): the
    # floor BC is disabled for benchmark runs, but staying clear of the
    # band keeps every proximity-gated branch in g2p provably inert.
    center_xz = 0.5 * (lo + hi)
    y_lift = max(2.0 * grid_dx, 0.05 + grid_dx)
    offset = np.array([-center_xz[0], -lo[1] + y_lift, -center_xz[2]])
    if thermal is not None and gravity_y < 0.0:
        # Standing-melt scene: the body rests ON the floor contact plane
        # (CONTACT_ZONE = 0.05) so the floor BC carries its weight.
        offset[1] = -bbox_lo[1] + 0.05

    mu, lam = lame_from_scene(scene["material"])
    density = float(scene["material"]["density_kg_m3"])
    particle_volume = spacing ** 3
    mass = density * particle_volume

    # Substep dt: CFL against the longitudinal wave speed on the grid,
    # with a healthy safety factor, capped by the frame dt.
    c_wave = float(np.sqrt((lam + 2.0 * mu) / density))
    dt_cfl = 0.3 * grid_dx / max(c_wave, 1e-9)
    dt_sub = float(args.substep_dt) if args.substep_dt else min(dt_cfl,
                                                                frame_dt)
    substeps = max(1, int(np.ceil(frame_dt / dt_sub)))
    dt_sub = frame_dt / substeps

    return {
        "gravity_y": gravity_y, "fps": fps, "duration": duration,
        "n_frames": n_frames, "frame_dt": frame_dt, "dt_sub": dt_sub,
        "substeps": substeps, "grid_dx": grid_dx, "grid_lim": grid_lim,
        "grid_coarsened": coarsened, "co_moving": co_moving,
        "offset": offset, "mu": mu, "lam": lam,
        "density": density, "mass": mass, "spacing": spacing,
        "n_particles": len(scene_pos),
    }


# ---------------------------------------------------------------------------
# Initial conditions (scene frame; positions untouched, velocities set)
# ---------------------------------------------------------------------------

def initial_velocities(scene: dict, scene_pos: np.ndarray) -> np.ndarray:
    ic = scene.get("initial_conditions", {})
    n = len(scene_pos)
    vel = np.zeros((n, 3))
    if "linear_velocity_m_s" in ic:
        vel += np.asarray(ic["linear_velocity_m_s"], float)[None, :]
    if "angular_velocity_rad_s" in ic:
        omega = np.asarray(ic["angular_velocity_rad_s"], float)
        x_cm = scene_pos.mean(axis=0)  # equal particle masses
        vel += np.cross(omega[None, :], scene_pos - x_cm[None, :])
    if ic.get("type") == "modal_velocity_pluck":
        # First cantilever mode shape, normalized to 1 at the tip, applied
        # along the scene's pluck direction. Formula from the scene JSON.
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
# Boundary condition pin schedules
# ---------------------------------------------------------------------------

class PinGroup:
    """A set of particles whose position is prescribed:
    pos(t) = base + alpha(t) * disp, vel(t) = alpha_rate(t) * disp.
    base and disp are sim-frame arrays over ALL particles (zero off-mask).

    Particle pins alone cannot enforce a displacement BC in MPM: the pinned
    layer's velocity is mass-diluted on the grid by the B-spline stencil
    (measured on A5: the interior received about 2 percent of the commanded
    strain, exponentially relaxing back). So each group also carries a
    GRID-level constraint: nodes inside the group's slab (the pinned
    particles' target bounding box, padded by one stencil reach) get their
    velocity overwritten with the prescribed affine field
        v(x, t) = rate(t) * (A0 @ x + b0)
    which covers every scene BC in the suite (clamps, face ramps, affine
    ramps, shear drives, plates, pulses). A0 and b0 are in sim coordinates.
    """

    def __init__(self, mask: np.ndarray, base_sim: np.ndarray,
                 disp_sim: np.ndarray, alpha_fn, rate_fn, active_fn=None,
                 A0: np.ndarray | None = None, b0: np.ndarray | None = None,
                 AB_fn=None):
        self.mask = mask
        self.base = base_sim
        self.disp = disp_sim
        self.alpha_fn = alpha_fn
        self.rate_fn = rate_fn
        self.active_fn = active_fn or (lambda t: True)
        self.A0 = np.zeros((3, 3)) if A0 is None else np.asarray(A0, float)
        self.b0 = np.zeros(3) if b0 is None else np.asarray(b0, float)
        # Grid field at time t: v(x) = A(t) @ x + b(t). The default scales
        # the static (A0, b0) by the schedule rate, which is exact when the
        # prescribed map does not change the coordinates it is expressed in
        # (clamps, translations, simple shear along an invariant axis).
        # Deformations that stretch their own coordinate (the affine ramp,
        # the plate compression) must supply AB_fn with the pull-back to
        # rest coordinates, otherwise the field is wrong by a factor of the
        # accumulated stretch at large alpha (measured 18 percent on C1).
        self.AB_fn = AB_fn or (lambda t: (self.rate_fn(t) * self.A0,
                                          self.rate_fn(t) * self.b0))
        # Slab bounds come from the pinned particles' rest positions and
        # their displacement extremes; recomputed per substep from alpha.
        self._pin_base = base_sim[mask]
        self._pin_disp = disp_sim[mask]

    def slab(self, alpha: float, pad: float):
        target = self._pin_base + alpha * self._pin_disp
        return target.min(axis=0) - pad, target.max(axis=0) + pad


def build_pins(scene: dict, scene_pos: np.ndarray,
               offset: np.ndarray, notes: list) -> list[PinGroup]:
    pins: list[PinGroup] = []
    base_sim = scene_pos + offset[None, :]
    bcs = scene.get("boundary_conditions", {})
    zero = np.zeros_like(scene_pos)

    def _static(mask):
        pins.append(PinGroup(mask, base_sim, zero,
                             lambda t: 0.0, lambda t: 0.0))  # A0=b0=0: v=0 slab

    for name, bc in bcs.items():
        if not isinstance(bc, dict):
            if name == "lateral_faces" and bc == "traction_free":
                notes.append(
                    "Lateral faces: traction-free honored. The body floats "
                    "in the grid interior with the floor BC disabled; the "
                    "grid boundary guard only zeroes weights at the domain "
                    "edge, which the body never approaches.")
            continue
        btype = bc.get("type", "")
        if btype == "fixed":
            _static(parse_region(bc["region"], scene_pos))
        elif btype == "displacement_ramp":
            mask = parse_region(bc["region"], scene_pos)
            dvec = np.asarray(bc["displacement_m"], float)
            disp = np.zeros_like(scene_pos)
            disp[mask] = dvec[None, :]
            T = float(bc["ramp_end_s"])
            pins.append(PinGroup(
                mask, base_sim, disp,
                lambda t, T=T: min(t / T, 1.0),
                lambda t, T=T: (1.0 / T) if t < T else 0.0,
                b0=dvec))
        elif btype == "displacement_ramp_affine":
            F_target = np.asarray(bc["F_target"], float)
            about = np.asarray(bc.get("about", [0, 0, 0]), float)
            # pin the boundary shell (within one spacing of the bbox faces)
            spacing = float(scene["geometry"]["particle_spacing_m"])
            lo = scene_pos.min(axis=0) + spacing * 0.75
            hi = scene_pos.max(axis=0) - spacing * 0.75
            mask = ~np.all((scene_pos > lo) & (scene_pos < hi), axis=1)
            A = F_target - np.eye(3)
            disp = (scene_pos - about) @ A.T
            disp[~mask] = 0.0
            T = float(bc["ramp_end_s"])
            alpha_fn = lambda t, T=T: min(t / T, 1.0)
            rate_fn = lambda t, T=T: (1.0 / T) if t < T else 0.0

            def _ab_affine(t, A=A, about=about, offset=offset,
                           alpha_fn=alpha_fn, rate_fn=rate_fn):
                # v(x0) = rate * A x0 in REST coordinates; pulled back from
                # current coordinates through M = I + alpha A:
                # x0 = M^-1 (x - offset - about) + about
                M_inv = np.linalg.inv(np.eye(3) + alpha_fn(t) * A)
                At = rate_fn(t) * (A @ M_inv)
                return At, -At @ (offset + about)

            pins.append(PinGroup(mask, base_sim, disp, alpha_fn, rate_fn,
                                 AB_fn=_ab_affine))
        elif btype == "prescribed_simple_shear":
            gamma = float(bc["gamma_peak"])
            spacing = float(scene["geometry"]["particle_spacing_m"])
            y = scene_pos[:, 1]
            mask = (y < y.min() + spacing * 0.75) \
                | (y > y.max() - spacing * 0.75)
            disp = np.zeros_like(scene_pos)
            disp[mask, 0] = gamma * y[mask]
            T = float(scene["simulation"]["duration_s"])
            A = np.zeros((3, 3))
            A[0, 1] = gamma          # v_x = rate * gamma * y_scene
            pins.append(PinGroup(
                mask, base_sim, disp,
                lambda t, T=T: 1.0 - abs(2.0 * t / T - 1.0),
                lambda t, T=T: (2.0 / T) if t < T / 2 else (-2.0 / T),
                A0=A, b0=np.array([-gamma * offset[1], 0.0, 0.0])))
        elif btype == "prescribed_displacement_pulse":
            mask = parse_region(bc["region"], scene_pos)
            avec = (float(bc["amplitude_m"])
                    * np.asarray(bc["direction"], float))
            disp = np.zeros_like(scene_pos)
            disp[mask] = avec[None, :]
            W = float(bc["width_s"])
            pins.append(PinGroup(
                mask, base_sim, disp,
                lambda t, W=W: np.sin(np.pi * t / W) if t <= W else 0.0,
                lambda t, W=W: ((np.pi / W) * np.cos(np.pi * t / W)
                                if t <= W else 0.0),
                b0=avec))
            notes.append(
                "Pulse face stays pinned at zero displacement after the "
                "pulse window (piston held still); arrival detection at the "
                "far end happens before any reflection returns.")
        elif btype == "displacement_controlled_plate":
            # Handled as a true unilateral plate (a moving plane that
            # constrains whatever is currently above it, y only), not a
            # rest-mask pin: a curved asset makes progressive contact with
            # a flat plate, and side material legitimately squeezes up
            # around a cap-only pin (measured: commanded 80 percent
            # compression read as 82.8 with a rest-mask band). See the
            # Plate wiring in main().
            notes.append(
                "Squash plate: unilateral moving plane, frictionless in "
                "the plate tangent (only v_y constrained), progressive "
                "contact, removed after hold_end_s "
                f"= {bc['hold_end_s']}.")
            continue
        elif btype == "fixed_temperature":
            # Handled by the per-frame heat solve (heat_step_physical
            # Dirichlet band), not by a mechanical pin.
            continue
        else:
            raise SystemExit(f"REFUSED: boundary type '{btype}' is not "
                             "supported by the exporter harness.")
    return pins


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tier-c", action="store_true",
                    help="Record per-frame covariances.")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="Duration override for dry runs.")
    ap.add_argument("--substep-dt", type=float, default=None,
                    help="Override the CFL-derived substep dt.")
    args = ap.parse_args(argv)

    scene, scene_pos_raw = load_scene(args.scene)
    thermal = scene.get("material", {}).get("thermal")
    # Thermal scenes are z-up; work in permuted (y-up) coordinates from
    # here on, and hand the permutation to the recorder for exact
    # inversion on export.
    PERM = (np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
            if thermal is not None else None)
    scene_pos = scene_pos_raw @ PERM.T if PERM is not None else scene_pos_raw
    plan = plan_run(scene, scene_pos, args)
    notes: list[str] = []
    use_floor = bool(thermal is not None and plan["gravity_y"] < 0.0)

    LOGGER.info(
        "[bench] scene %s: %s particles, %s frames at %.0f fps, substep dt %.2e s x %s/frame",
        scene["scene_id"],
        plan["n_particles"],
        plan["n_frames"],
        plan["fps"],
        plan["dt_sub"],
        plan["substeps"],
    )
    LOGGER.info(
        "[bench] grid: dx=%.4f m, lim=%.2f m%s",
        plan["grid_dx"],
        plan["grid_lim"],
        " (COARSENED to fit the node cap; grid resolution is the submitter's choice, declared in notes)"
        if plan["grid_coarsened"] else "",
    )
    if plan["grid_coarsened"]:
        notes.append(
            f"Grid dx coarsened to {plan['grid_dx']:.4f} m so the "
            f"{plan['grid_lim']:.1f} m motion envelope fits the solver's "
            "150^3 node cap.")

    # Seed everything BEFORE importing the simulator (its module body
    # samples shape point clouds with numpy RNG).
    seed = 0
    np.random.seed(seed)

    # Configure the simulator via its own CLI, then import it. All grid,
    # timestep, and gravity constants derive from these flags at import.
    sys.argv = [
        "gaussian_mesh_poc.py", "--headless", "--no-skin",
        "--shape", "sphere",
        "--n-per-entity", str(plan["n_particles"]),
        "--material", "1",
        "--gravity-y", str(plan["gravity_y"]),
        "--frame-dt", str(plan["frame_dt"]),
        "--substep-dt", str(plan["dt_sub"]),
        "--grid-dx", str(plan["grid_dx"]),
        "--grid-lim", str(plan["grid_lim"]),
    ]
    sys.path.insert(0, str(REPO))
    import gaussian_mesh_poc as G  # noqa: E402
    import solver_ul  # noqa: E402  (same module instance the sim uses)
    import taichi as ti  # noqa: E402

    sim = G.GaussianBallSim(material_key="1")
    sim.drop()

    # ---- overwrite the entity state with the scene's frozen particles ----
    n = plan["n_particles"]
    sl = slice(sim.start, sim.end)
    sim_pos = (scene_pos + plan["offset"][None, :]).astype(np.float32)

    G.REST_POS[sl] = sim_pos          # rest = spawn (spawn_position folded in)
    sim.spawn_position = np.zeros(3)
    sim.positions = sim_pos.astype(float)
    vel0 = initial_velocities(scene, scene_pos).astype(float)
    # Co-moving (Galilean-boosted) frame: simulate in the frame translating
    # at the scene's uniform drift velocity, so the body spins in place near
    # the coordinate origin. Physics is Galilean-invariant, so this is
    # exact; what it buys is numerical: at f32, position rounding grows with
    # |x|, and by 11.5 m of drift it reaches about 1 percent of the
    # per-substep rotational displacement, which measurably scrambles
    # angular momentum. The recorder applies the exact inverse transform on
    # export (x + v0 t, v + v0), so the submission is in scene coordinates.
    boost = np.zeros(3)
    if plan["co_moving"]:
        boost = np.asarray(
            scene["initial_conditions"]["linear_velocity_m_s"], float)
        vel0 = vel0 - boost[None, :]
    sim.velocities = vel0
    sim.Fs = np.tile(np.eye(3), (n, 1, 1))
    # APIC affine matrix seeded with the IC velocity gradient. For a rigid
    # rotation, grad(v) = skew(omega); leaving C = 0 misrepresents the
    # initial state (point velocities with zero local spin) and the first
    # P2G transfer then low-pass filters the rotational field, permanently
    # losing about (grid_dx / body_radius)^2 of the angular momentum.
    # Measured before this fix on a1a: 16.4 percent L loss in frame 1 with
    # dx/R = 0.4, matching (dx/R)^2 = 0.16. Uniform-velocity ICs have zero
    # gradient, so this is exactly C = 0 for them.
    sim.Cs = np.zeros((n, 3, 3))
    ic = scene.get("initial_conditions", {})
    if "angular_velocity_rad_s" in ic:
        wx, wy, wz = np.asarray(ic["angular_velocity_rad_s"], float)
        skew = np.array([[0.0, -wz, wy],
                         [wz, 0.0, -wx],
                         [-wy, wx, 0.0]])
        sim.Cs = np.tile(skew, (n, 1, 1))
        notes.append(
            "APIC affine matrix C seeded with skew(omega) to represent the "
            "rigid-rotation initial condition exactly (C = 0 would filter "
            "the rotational field on the first transfer).")
    sim.Fp = np.tile(np.eye(3), (n, 1, 1))
    sim._upload_to_gpu()

    def _set(field, arr):
        cur = field.to_numpy()
        cur[sl] = arr
        field.from_numpy(cur)

    _set(G._ti_rest, sim_pos)
    # Material: the one permitted conversion, plus honest per-particle
    # bookkeeping (mass from density * spacing^3, damping off, plasticity
    # off, isotropic rest covariance at half the particle spacing).
    _set(G._ti_mu, np.full(n, plan["mu"], dtype=np.float32))
    _set(G._ti_mu0, np.full(n, plan["mu"], dtype=np.float32))
    _set(G._ti_lam, np.full(n, plan["lam"], dtype=np.float32))
    _set(G._ti_m, np.full(n, plan["mass"], dtype=np.float32))
    # Per-particle rest volume: p2g scales internal forces by V0, so a stale
    # default here (the 1 m demo ball's volume) blows the run up instantly.
    _set(G._ti_v0, np.full(n, plan["spacing"] ** 3, dtype=np.float32))
    _set(G._ti_damping, np.zeros(n, dtype=np.float32))
    _set(G._ti_yield, np.full(n, 1e9, dtype=np.float32))
    _set(G._ti_relax, np.zeros(n, dtype=np.float32))
    _set(G._ti_plastic_j2, np.zeros(n, dtype=np.int32))
    _set(G._ti_model, np.zeros(n, dtype=np.int32))   # corotated
    sigma0 = (plan["spacing"] * 0.5) ** 2
    rcov = np.tile((np.eye(3) * sigma0)[None], (n, 1, 1)).astype(np.float32)
    _set(G._ti_rcov, rcov)
    sim.rest_covs = rcov.astype(float)
    # sim.covs feeds the frame-0 covariance export; without this it would
    # hold the stale demo-shape covariances until the first _finish_frame.
    sim.covs = rcov.astype(float)
    _set(G._ti_covs, rcov)

    # Thermal state + calibration (SI-exact heat path, see
    # solver_ul.heat_step_physical). Molten-phase mechanical parameters are
    # not specified by the scenes; wax-like defaults are declared.
    MOLTEN_KAPPA, MOLTEN_MU = 4.0e5, 3.0e3
    heat_floor_y = float(sim_pos[:, 1].min())
    hot_temp = 0.0
    heat_every = 1
    if thermal is not None:
        # Batch size chosen so the roundtrip smoothing D_num stays a small,
        # compensable fraction of the physical diffusivity (see the loop).
        alpha_si_plan = (float(thermal["conductivity_w_m_k"])
                         / (plan["density"]
                            * float(thermal["specific_heat_j_kg_k"])))
        d_num_1 = plan["grid_dx"] ** 2 / (4.0 * plan["frame_dt"])
        heat_every = max(1, int(np.ceil(5.0 * d_num_1
                                        / max(alpha_si_plan, 1e-12))))
        t_init = float(scene["initial_conditions"].get(
            "uniform_temperature_k", thermal["initial_temperature_k"]))
        _set(G._ti_temp, np.full(n, t_init, dtype=np.float32))
        _set(G._ti_heat_U, np.zeros(n, dtype=np.float32))
        _set(G._ti_phase, np.zeros(n, dtype=np.int32))
        hot_temp = float(
            scene["boundary_conditions"]["hot_floor"]["temperature_k"])
        notes.append(
            "Thermal path: solver_ul.heat_step_physical, SI-exact explicit "
            "diffusion (alpha = k/(rho c) integrated over each frame in "
            "stability-bounded sub-iterations); Kelvin used directly (all "
            "operations affine in T). Dirichlet hot-floor band: 1 grid "
            "cell above the body base. Latent heat and melt temperature "
            "taken from the scene in SI. DEVIATION: molten-phase "
            f"mechanical response (kappa={MOLTEN_KAPPA:g} Pa, "
            f"mu={MOLTEN_MU:g} Pa) is not specified by the scene; wax-like "
            "defaults used. Scene axes are z-up; simulated permuted to "
            "y-up, exact inverse on export. Heat updates batched every "
            f"{heat_every} frames: each particle-grid temperature "
            "roundtrip adds B-spline smoothing diffusion dx^2/(4 dt); "
            "batching shrinks it and the residual is subtracted from the "
            "solve conductivity so total effective diffusivity is the "
            "scene's k/(rho c).")

    mu_in = scene["material"]["youngs_modulus_pa"]
    nu_in = scene["material"]["poisson_ratio"]
    notes.append(
        f"Material mapping: E={mu_in:g} Pa, nu={nu_in:g} converted to "
        f"mu={plan['mu']:.6g} Pa, lambda={plan['lam']:.6g} Pa "
        "(mu = E/(2(1+nu)), lambda = E nu/((1+nu)(1-2nu))). Density "
        f"{plan['density']:g} kg/m^3, per-particle mass "
        f"{plan['mass']:.6g} kg from spacing^3 volume.")
    notes.append(
        f"Solver path: UL-MLS-MPM (solver_ul.substep, use_floor=False). "
        f"frame_dt={plan['frame_dt']:.6g} s, substep_dt={plan['dt_sub']:.6g} "
        f"s, {plan['substeps']} substeps/frame. Floor contact BC disabled: "
        "benchmark scenes have no floor.")
    notes.append("Taichi arch=cuda, default_fp=f32 (single precision). "
                 f"numpy seed={seed}.")
    notes.append("Fixed/prescribed boundary regions are enforced by pin "
                 "kernels in the export module (position and velocity "
                 "overrides after each substep), not by grid BCs.")

    # ---- pins ------------------------------------------------------------
    pins = build_pins(scene, scene_pos, plan["offset"], notes)

    @ti.kernel
    def _apply_pin(start: ti.i32, mask: ti.types.ndarray(),
                   base: ti.types.ndarray(), disp: ti.types.ndarray(),
                   alpha: ti.f32, rate: ti.f32):
        for p in range(mask.shape[0]):
            if mask[p] == 1:
                q = start + p
                for d in ti.static(range(3)):
                    G._ti_pos[q][d] = base[p, d] + alpha * disp[p, d]
                    G._ti_vel[q][d] = rate * disp[p, d]
                G._ti_C[q] = ti.Matrix.zero(ti.f32, 3, 3)

    @ti.kernel
    def _apply_slab_bc(lo: ti.types.ndarray(), hi: ti.types.ndarray(),
                       A: ti.types.ndarray(), b: ti.types.ndarray(),
                       rate: ti.f32):
        # Overwrite grid velocities inside the slab with the prescribed
        # affine field v = rate * (A @ x + b). Same mechanism as the
        # solver's own floor BC, driven by the scene's boundary schedule.
        dx = solver_ul._UL_GRID_DX
        N = solver_ul._UL_GRID_N
        origin = solver_ul._ti_grid_origin[None]
        for i in range(solver_ul._MAX_GRID_UL):
            iz = i % N
            iy = (i // N) % N
            ix = i // (N * N)
            px = origin[0] + ix * dx
            py = origin[1] + iy * dx
            pz = origin[2] + iz * dx
            if (lo[0] <= px <= hi[0] and lo[1] <= py <= hi[1]
                    and lo[2] <= pz <= hi[2]):
                for r in ti.static(range(3)):
                    solver_ul._ti_gv_ul[i][r] = rate * (
                        A[r, 0] * px + A[r, 1] * py + A[r, 2] * pz + b[r])

    # Unilateral plate (c3): plane descends from the body top to
    # frac * height over press_end_s, holds, then vanishes. Constrains
    # only what is currently above it, only in y.
    plate = None
    bc_plate = scene.get("boundary_conditions", {}).get("squash_plate")
    if isinstance(bc_plate, dict) \
            and bc_plate.get("type") == "displacement_controlled_plate":
        y_sim = sim_pos[:, 1]
        h0 = float(y_sim.max() - y_sim.min())
        drop = (1.0 - float(bc_plate["target_height_frac"])) * h0
        press = float(bc_plate["press_end_s"])
        plate = {
            "top": float(y_sim.max()),
            "drop": drop,
            "press": press,
            "hold_end": float(bc_plate["hold_end_s"]),
            "speed": -drop / press,
        }

        # The scene's floor is a plane, not just the pinned bottom layer:
        # without it, squeezed material extrudes below y = floor (measured
        # 2.7 mm at full press, misread as missing compression).
        floor_plane = float(y_sim.min())

        @ti.kernel
        def _apply_plate_particles(start: ti.i32, end: ti.i32,
                                   plane: ti.f32, vplate: ti.f32,
                                   fl: ti.f32):
            for p in range(start, end):
                if G._ti_pos[p][1] > plane:
                    G._ti_pos[p][1] = plane
                    if G._ti_vel[p][1] > vplate:
                        G._ti_vel[p][1] = vplate
                if G._ti_pos[p][1] < fl:
                    G._ti_pos[p][1] = fl
                    if G._ti_vel[p][1] < 0.0:
                        G._ti_vel[p][1] = 0.0

        @ti.kernel
        def _apply_plate_grid(plane: ti.f32, vplate: ti.f32, fl: ti.f32):
            dx = solver_ul._UL_GRID_DX
            N = solver_ul._UL_GRID_N
            origin = solver_ul._ti_grid_origin[None]
            for i in range(solver_ul._MAX_GRID_UL):
                iy = (i // N) % N
                py = origin[1] + iy * dx
                if py > plane - 0.75 * dx:
                    if solver_ul._ti_gv_ul[i][1] > vplate:
                        solver_ul._ti_gv_ul[i][1] = vplate
                if py < fl + 0.75 * dx:
                    if solver_ul._ti_gv_ul[i][1] < 0.0:
                        solver_ul._ti_gv_ul[i][1] = 0.0

        def plate_state(t):
            if t > plate["hold_end"]:
                return None
            a = min(t / plate["press"], 1.0)
            plane = plate["top"] - a * plate["drop"]
            v = plate["speed"] if t < plate["press"] else 0.0
            return plane, v

    pin_bufs = [(g, g.mask.astype(np.int32),
                 g.base.astype(np.float32), g.disp.astype(np.float32))
                for g in pins]
    slab_pad = 0.75 * plan["grid_dx"]

    def apply_grid_bcs(t):
        # Runs INSIDE the substep (via solver_ul.substep grid_bc_fn), after
        # the grid update and before g2p, so the constraint shapes what the
        # particles gather. Particle pins alone are mass-diluted by the
        # stencil and transmit almost nothing (measured: 2 percent of the
        # commanded strain on A5).
        for g, m, bb, d in pin_bufs:
            if g.active_fn(t):
                lo, hi = g.slab(g.alpha_fn(t), slab_pad)
                At, bt = g.AB_fn(t)
                _apply_slab_bc(lo.astype(np.float32), hi.astype(np.float32),
                               At.astype(np.float32), bt.astype(np.float32),
                               1.0)
        if plate is not None:
            ps = plate_state(t)
            if ps is not None:
                _apply_plate_grid(float(ps[0]), float(ps[1]),
                                  float(floor_plane))

    def apply_pins(t):
        for g, m, bb, d in pin_bufs:
            if g.active_fn(t):
                _apply_pin(sim.start, m, bb, d,
                           float(g.alpha_fn(t)), float(g.rate_fn(t)))
        if plate is not None:
            ps = plate_state(t)
            if ps is not None:
                _apply_plate_particles(sim.start, sim.end,
                                       float(ps[0]), float(ps[1]),
                                       float(floor_plane))

    # Pin at t=0 so frame 0 velocities are consistent with the pins
    # (for example the clamped beam root starts at zero velocity even
    # though the modal pluck formula is nonzero there).
    if pins:
        v = sim.velocities.copy()
        for g in pins:
            v[g.mask] = g.rate_fn(0.0) * g.disp[g.mask]
        sim.velocities = v
        sim._upload_to_gpu()

    # ---- recorder ----------------------------------------------------------
    from export.benchexport import BenchRecorder
    rec = BenchRecorder(
        args.out / scene["scene_id"], args.scene,
        record_velocities=True, record_mass=True,
        record_phase=thermal is not None,
        record_covariances=bool(args.tier_c),
        scene_to_sim={"offset": plan["offset"].tolist(),
                      "permutation":
                          PERM.tolist() if PERM is not None else None,
                      "frame_velocity":
                          boost.tolist() if plan["co_moving"] else None},
        notes=notes)
    # C3 renders: the round-trip scene needs before/after images from OUR
    # renderer (the contract under test is the physics-to-appearance link).
    # The procedural asset gets a synthesized per-particle color texture
    # (deterministic sinusoidal field) so SSIM has spatial structure to
    # compare; a uniform color would only test the silhouette.
    gs_snap = None
    if scene["scene_id"].startswith("c3_"):
        from gs_render import GSRenderer
        rgb = 0.5 + 0.35 * np.sin(
            scene_pos @ np.array([[41.0, 13.0, 7.0],
                                  [11.0, 47.0, 17.0],
                                  [5.0, 19.0, 53.0]]).T)
        shs = np.zeros((n, 16, 3), dtype=np.float32)
        shs[:, 0, :] = ((rgb - 0.5) / 0.28209479).astype(np.float32)
        opac = np.full((n, 1), 0.85, dtype=np.float32)
        gs = GSRenderer(opac, shs, bg=(1.0, 1.0, 1.0))
        ctr = sim_pos.mean(axis=0).astype(np.float64)
        rad = float(np.linalg.norm(sim_pos - ctr, axis=1).max())
        eye = ctr + np.array([0.35, 0.3, 0.9]) * (3.2 * rad)
        up = np.array([0.0, 1.0, 0.0])

        def gs_snap(name):
            img = gs.render(sim.positions.astype(np.float32),
                            sim.covs.astype(np.float32),
                            eye.astype(np.float32), ctr.astype(np.float32),
                            up.astype(np.float32), 45.0, 512, 512)
            rec.add_render(0, img, name=name)

        rec.write_camera({
            "position": (eye - plan["offset"]).tolist(),
            "look_at": (ctr - plan["offset"]).tolist(),
            "up": up.tolist(), "fov_deg": 45.0,
            "width": 512, "height": 512,
            "note": "scene coordinates; renders produced by gs_render "
                    "(diff-gaussian-rasterization) from the simulator's "
                    "own deformed covariances"})
        gs_snap("before.png")

    if args.max_seconds is not None:
        rec.notes.append(
            f"DEVIATION: duration truncated to {plan['duration']:g} s by "
            "--max-seconds (dry run, not a scoreable submission).")

    # ---- main loop ---------------------------------------------------------
    def state():
        st = {"positions": sim.positions, "velocities": sim.velocities,
              "mass": np.full(n, plan["mass"], dtype=np.float32)}
        if args.tier_c:
            st["covariances"] = sim.covs
        if thermal is not None:
            st["phase"] = G._ti_phase.to_numpy()[sl]
        return st

    if plan["co_moving"]:
        rec.notes.append(
            "Co-moving frame (Galilean boost): simulated in the frame "
            f"translating at v_frame={boost.tolist()} m/s, so the body "
            "stays near the coordinate origin and the grid needs only the "
            f"body extent (dx {plan['grid_dx']:.4g} m) instead of the full "
            "travel envelope. Exact inverse (x + v_frame t, v + v_frame) "
            "applied on export.")

    kappa = 0.0
    t_report = max(1, plan["n_frames"] // 10)
    rec.on_frame(state(), 0, 0.0)
    for frame in range(1, plan["n_frames"]):
        t0 = (frame - 1) * plan["frame_dt"]
        for s in range(plan["substeps"]):
            t_sub = t0 + (s + 1) * plan["dt_sub"]
            solver_ul.substep(sim.start, sim.end, plan["dt_sub"],
                              kappa=kappa, mu_scale=1.0, model=0,
                              floor_damp=0.0, friction=0.0,
                              use_floor=use_floor,
                              grid_bc_fn=(lambda t=t_sub: apply_grid_bcs(t))
                              if (pins or plate is not None) else None)
            apply_pins(t_sub)
        if thermal is not None and frame % heat_every == 0:
            # Batched heat updates: every particle-grid temperature
            # roundtrip convolves T with the B-spline kernel, adding
            # numerical diffusion D_num = dx^2 / (4 dt_batch) on top of the
            # physical alpha (measured on A4 as a 3x-too-fast melt front
            # when run per frame, where D_num was 5x alpha). Batching K
            # frames divides D_num by K, and the residual is compensated by
            # solving with k_eff = (alpha - D_num) rho c, so the total
            # effective diffusivity matches the scene's SI value.
            dt_batch = heat_every * plan["frame_dt"]
            alpha_si = (float(thermal["conductivity_w_m_k"])
                        / (plan["density"]
                           * float(thermal["specific_heat_j_kg_k"])))
            d_num = plan["grid_dx"] ** 2 / (4.0 * dt_batch)
            k_eff = ((alpha_si - d_num) * plan["density"]
                     * float(thermal["specific_heat_j_kg_k"]))
            solver_ul.heat_step_physical(
                sim.start, sim.end, dt_batch,
                conductivity=k_eff,
                heat_capacity=float(thermal["specific_heat_j_kg_k"]),
                density=plan["density"],
                floor_y=heat_floor_y, source_temp=hot_temp, band_dx=1.0)
            G._ti_phase_update(
                sim.start, sim.end,
                float(thermal["melt_temperature_k"]),
                float(thermal["latent_heat_j_kg"]),
                MOLTEN_KAPPA,
                float(thermal["specific_heat_j_kg_k"]),
                MOLTEN_MU)
        sim._finish_frame()
        rec.on_frame(state(), frame, frame * plan["frame_dt"])
        if frame % t_report == 0:
            LOGGER.info("[bench] frame %s/%s", frame, plan["n_frames"] - 1)
    if gs_snap is not None:
        gs_snap("after.png")

    extra_meta = {"scene_id": scene["scene_id"]}
    # Scenes may require structured declarations in meta.json (SPEC 2.2.1).
    # The only one implemented is domain_handling: GaussianFlesh's UL world
    # grid is a fixed dense allocation sized before the run, so the honest
    # category is "fixed". When notes becomes an object, the free-text
    # deviation notes move to notes.text per the spec.
    req = scene.get("pass_criteria", {}).get("requires_meta_declaration", {})
    if req:
        unknown = [k for k in req if k != "notes.domain_handling"]
        if unknown:
            raise SystemExit(f"REFUSED: scene requires declarations "
                             f"{unknown} the exporter does not know how "
                             "to make truthfully.")
        strategy = "co_moving" if plan["co_moving"] else "fixed"
        extra_meta["notes"] = {"domain_handling": strategy,
                               "text": " | ".join(rec.notes)}
    out = rec.finalize(extra_meta=extra_meta)
    LOGGER.info("[bench] submission written to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
