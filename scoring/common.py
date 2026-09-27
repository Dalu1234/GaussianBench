"""Shared infrastructure for GaussianBench scorers.

Everything a scorer needs to read a scene JSON and a trajectory submission,
validate it strictly, and emit a verdict. No simulator imports, ever.
Dependencies: numpy, pandas, pyarrow (through pandas.read_parquet).

Design contract (see SPEC.md):
  - Scorers are pure functions of files.
  - Tolerances live in scene JSON pass_criteria, never in scorer code.
  - NA and FAIL are distinct outcomes. INVALID means a regime assumption was
    violated so the analytical reference does not apply.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

SPEC_VERSION = "1.2.0"

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_NA = "NA"
STATUS_INVALID = "INVALID"
STATUS_ERROR = "ERROR"

VALID_TEST_CLASSES = (
    "code_verification",
    "solution_verification",
    "model_validation",
    "capability",
)


class SubmissionError(Exception):
    """Raised when a submission file is missing, malformed, or violates SPEC.md.

    The message is shown verbatim to the submitter, so it must say exactly
    what is wrong and how to fix it.
    """


# ---------------------------------------------------------------------------
# Scene loading
# ---------------------------------------------------------------------------

def load_scene(scene_path: str | Path) -> dict:
    """Load and sanity-check a scene JSON. Returns the parsed dict."""
    scene_path = Path(scene_path)
    if not scene_path.exists():
        raise SubmissionError(f"Scene file not found: {scene_path}")
    with open(scene_path, "r", encoding="utf-8") as f:
        scene = json.load(f)
    for key in ("scene_id", "test_class", "tier", "simulation", "pass_criteria"):
        if key not in scene:
            raise SubmissionError(
                f"Scene {scene_path.name} is missing required key '{key}'. "
                "Scene files must be fully self-describing, see SPEC.md section 3.")
    if scene["test_class"] not in VALID_TEST_CLASSES:
        raise SubmissionError(
            f"Scene {scene['scene_id']} has unknown test_class "
            f"'{scene['test_class']}'. Must be one of {VALID_TEST_CLASSES}.")
    return scene


def load_scene_positions(scene: dict, scenes_dir: str | Path) -> np.ndarray:
    """Load the frozen initial particle positions (N, 3) for a scene."""
    fname = scene.get("geometry", {}).get("particle_positions_file")
    if fname is None:
        raise SubmissionError(
            f"Scene {scene['scene_id']} declares no particle_positions_file.")
    path = Path(scenes_dir) / fname
    if not path.exists():
        raise SubmissionError(f"Scene positions file not found: {path}")
    pts = np.load(path)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise SubmissionError(
            f"{path.name} must be float array of shape (N, 3), got {pts.shape}.")
    return pts.astype(np.float64)


# ---------------------------------------------------------------------------
# Trajectory loading and validation
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Trajectory:
    """A validated trajectory in dense array form.

    positions:  (F, N, 3) float64, meters. Particle axis is sorted by particle_id
                and aligned with the scene positions file row order.
    times:      (F,) float64 seconds, strictly increasing.
    velocities: (F, N, 3) float64 or None if the submission omitted them.
    masses:     (N,) float64 or None.
    phase:      (F, N) int8 or None. 0 solid, 1 fluid.
    velocities_derived: True when velocities were reconstructed by finite
                differences because the submission had none. Scorers must
                then use the looser derived_velocity tolerance tier.
    """

    positions: np.ndarray
    times: np.ndarray
    velocities: Optional[np.ndarray]
    masses: Optional[np.ndarray]
    phase: Optional[np.ndarray]
    particle_ids: np.ndarray
    velocities_derived: bool = False

    @property
    def n_frames(self) -> int:
        return self.positions.shape[0]

    @property
    def n_particles(self) -> int:
        return self.positions.shape[1]

    def ensure_velocities(self) -> "Trajectory":
        """Return self, deriving velocities by central differences if absent.

        Central differences on the interior, one-sided at the ends. This is
        second-order accurate on the interior, matching what a black box that
        only outputs positions can be graded on.
        """
        if self.velocities is not None:
            return self
        v = np.empty_like(self.positions)
        t = self.times
        dt_fwd = t[1:] - t[:-1]
        if np.any(dt_fwd <= 0):
            raise SubmissionError("time must be strictly increasing.")
        # interior: (x[k+1] - x[k-1]) / (t[k+1] - t[k-1])
        v[1:-1] = (self.positions[2:] - self.positions[:-2]) / (
            (t[2:] - t[:-2])[:, None, None])
        v[0] = (self.positions[1] - self.positions[0]) / dt_fwd[0]
        v[-1] = (self.positions[-1] - self.positions[-2]) / dt_fwd[-1]
        return dataclasses.replace(self, velocities=v, velocities_derived=True)


REQUIRED_COLUMNS = ("frame", "time", "particle_id", "x", "y", "z")
OPTIONAL_VELOCITY = ("vx", "vy", "vz")


def _read_trajectory_table(results_dir: Path) -> pd.DataFrame:
    pq = results_dir / "trajectory.parquet"
    csv = results_dir / "trajectory.csv"
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception as e:  # pyarrow raises several types
            raise SubmissionError(f"Could not read {pq}: {e}") from e
    if csv.exists():
        try:
            return pd.read_csv(csv)
        except Exception as e:
            raise SubmissionError(f"Could not read {csv}: {e}") from e
    raise SubmissionError(
        f"No trajectory.parquet or trajectory.csv in {results_dir}. "
        "See SPEC.md section 2.1 for the submission layout.")


def load_trajectory(results_dir: str | Path) -> Trajectory:
    """Load and strictly validate a trajectory submission directory.

    Raises SubmissionError with a fix-it message on any violation of SPEC.md.
    """
    results_dir = Path(results_dir)
    df = _read_trajectory_table(results_dir)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SubmissionError(
            f"trajectory is missing required columns {missing}. "
            f"Required: {list(REQUIRED_COLUMNS)}. Present: {list(df.columns)}.")

    if len(df) == 0:
        raise SubmissionError("trajectory contains zero rows.")

    for c in ("x", "y", "z"):
        col = df[c].to_numpy()
        if not np.all(np.isfinite(col)):
            bad = int(np.count_nonzero(~np.isfinite(col)))
            raise SubmissionError(
                f"trajectory column '{c}' contains {bad} non-finite values "
                "(NaN or inf). The simulation likely blew up; do not submit "
                "exploded runs.")

    frames = np.sort(df["frame"].unique())
    if frames[0] != 0:
        raise SubmissionError(
            f"First frame must be 0 (the initial condition), got {frames[0]}.")
    if not np.array_equal(frames, np.arange(len(frames))):
        raise SubmissionError(
            "Frame indices must be contiguous integers starting at 0. "
            f"Found gaps; first few frames present: {frames[:10].tolist()}.")

    # Per-frame particle set consistency and time consistency.
    ids0 = np.sort(df.loc[df["frame"] == 0, "particle_id"].unique())
    n = len(ids0)
    counts = df.groupby("frame", sort=True).size().to_numpy()
    if not np.all(counts == n):
        bad_frame = int(np.argmax(counts != n))
        raise SubmissionError(
            f"Frame {bad_frame} has {counts[bad_frame]} rows but frame 0 has {n}. "
            "Every frame must contain exactly the same particle ids "
            "(no birth or death within a run).")

    tpf = df.groupby("frame", sort=True)["time"].agg(["min", "max"])
    if np.any(tpf["min"].to_numpy() != tpf["max"].to_numpy()):
        raise SubmissionError(
            "All particles in a frame must share the same 'time' value.")
    times = tpf["min"].to_numpy().astype(np.float64)
    if np.any(np.diff(times) <= 0):
        k = int(np.argmax(np.diff(times) <= 0))
        raise SubmissionError(
            f"time must be strictly increasing with frame; violated at frame {k + 1} "
            f"(t={times[k]} then t={times[k + 1]}).")

    # Dense pivot: sort by (frame, particle_id) and reshape.
    df = df.sort_values(["frame", "particle_id"], kind="mergesort")
    ids_all = df["particle_id"].to_numpy().reshape(len(frames), n)
    if not np.all(ids_all == ids0[None, :]):
        raise SubmissionError(
            "particle_id sets differ between frames. Ids must be stable: the "
            "same physical particle keeps the same id in every frame.")

    positions = (df[["x", "y", "z"]].to_numpy()
                 .reshape(len(frames), n, 3).astype(np.float64))

    velocities = None
    if all(c in df.columns for c in OPTIONAL_VELOCITY):
        vel = df[list(OPTIONAL_VELOCITY)].to_numpy()
        if not np.all(np.isfinite(vel)):
            raise SubmissionError(
                "velocity columns contain non-finite values (NaN or inf).")
        velocities = vel.reshape(len(frames), n, 3).astype(np.float64)

    masses = None
    if "mass" in df.columns:
        m0 = df.loc[df["frame"] == 0].sort_values("particle_id")["mass"].to_numpy()
        per_frame = df["mass"].to_numpy().reshape(len(frames), n)
        if not np.allclose(per_frame, m0[None, :], rtol=0, atol=0):
            raise SubmissionError(
                "mass must be constant per particle across all frames.")
        if np.any(~np.isfinite(m0)) or np.any(m0 <= 0):
            raise SubmissionError("mass values must be finite and positive.")
        masses = m0.astype(np.float64)

    phase = None
    if "phase" in df.columns:
        ph = df["phase"].to_numpy().reshape(len(frames), n)
        if not np.all(np.isin(ph, (0, 1))):
            raise SubmissionError("phase must be 0 (solid) or 1 (fluid).")
        phase = ph.astype(np.int8)

    return Trajectory(
        positions=positions, times=times, velocities=velocities,
        masses=masses, phase=phase, particle_ids=ids0)


def check_initial_condition(traj: Trajectory, scene_positions: np.ndarray,
                            atol: float = 1e-6) -> None:
    """Verify frame 0 matches the frozen scene positions, row for row."""
    if traj.n_particles != scene_positions.shape[0]:
        raise SubmissionError(
            f"Submission has {traj.n_particles} particles but the scene "
            f"positions file has {scene_positions.shape[0]}. Every entrant "
            "must start from literally the shipped initial positions.")
    err = np.abs(traj.positions[0] - scene_positions).max()
    if err > atol:
        raise SubmissionError(
            f"Frame 0 deviates from the scene positions file by up to {err:.3e} m "
            f"(allowed {atol:.1e}). particle_id i must correspond to row i of "
            "the scene .npy, unpermuted and unscaled.")


def check_output_fps(traj: Trajectory, scene: dict, rel_tol: float = 0.01) -> None:
    """Verify the submission's output sampling matches simulation.output_fps."""
    fps_spec = float(scene["simulation"]["output_fps"])
    dt = np.diff(traj.times)
    fps_meas = 1.0 / float(np.median(dt))
    if abs(fps_meas - fps_spec) / fps_spec > rel_tol:
        raise SubmissionError(
            f"Output frame rate is {fps_meas:.2f} fps but the scene requires "
            f"{fps_spec:.2f} fps (within {rel_tol:.0%}). Internal substepping is "
            "free, output sampling is not.")


def infer_masses(traj: Trajectory, scene: dict) -> tuple[np.ndarray, bool]:
    """Return per-particle masses and whether they were assumed (not submitted).

    When the submission has no mass column, SPEC.md prescribes equal masses
    summing to density * geometry volume.
    """
    if traj.masses is not None:
        return traj.masses, False
    density = float(scene["material"]["density_kg_m3"])
    geom = scene["geometry"]
    if geom["type"] == "box":
        vol = float(np.prod(geom["size_m"]))
    elif geom["type"] == "sphere":
        vol = 4.0 / 3.0 * np.pi * float(geom["radius_m"]) ** 3
    else:
        raise SubmissionError(
            f"Cannot infer particle masses for geometry type '{geom['type']}'. "
            "Submit a mass column.")
    m = np.full(traj.n_particles, density * vol / traj.n_particles)
    return m, True


# ---------------------------------------------------------------------------
# Covariance loading (Tier C)
# ---------------------------------------------------------------------------

COV_COLUMNS = ("sxx", "sxy", "sxz", "syy", "syz", "szz")


def load_covariances(results_dir: str | Path,
                     frames: Optional[list[int]] = None) -> dict[int, np.ndarray]:
    """Load covariances.parquet as {frame: (N, 3, 3) array}.

    Particle axis is sorted by particle_id. If frames is given, only those
    frames are returned (all must be present).
    """
    path = Path(results_dir) / "covariances.parquet"
    if not path.exists():
        raise SubmissionError(
            f"No covariances.parquet in {results_dir}. This scene is Tier C "
            "and requires per-particle covariances, see SPEC.md section 2.3.")
    df = pd.read_parquet(path)
    missing = [c for c in ("frame", "particle_id") + COV_COLUMNS
               if c not in df.columns]
    if missing:
        raise SubmissionError(
            f"covariances.parquet missing columns {missing}. "
            f"Required: frame, particle_id, {', '.join(COV_COLUMNS)}.")
    vals = df[list(COV_COLUMNS)].to_numpy()
    if not np.all(np.isfinite(vals)):
        raise SubmissionError("covariances contain non-finite values.")

    out: dict[int, np.ndarray] = {}
    want = set(frames) if frames is not None else set(df["frame"].unique())
    for fr in sorted(want):
        sub = df[df["frame"] == fr]
        if len(sub) == 0:
            raise SubmissionError(
                f"covariances.parquet has no rows for required frame {fr}.")
        sub = sub.sort_values("particle_id")
        sxx, sxy, sxz, syy, syz, szz = (sub[c].to_numpy() for c in COV_COLUMNS)
        n = len(sub)
        cov = np.empty((n, 3, 3))
        cov[:, 0, 0] = sxx; cov[:, 0, 1] = sxy; cov[:, 0, 2] = sxz
        cov[:, 1, 0] = sxy; cov[:, 1, 1] = syy; cov[:, 1, 2] = syz
        cov[:, 2, 0] = sxz; cov[:, 2, 1] = syz; cov[:, 2, 2] = szz
        out[int(fr)] = cov
    return out


def load_deformation_gradients(results_dir: str | Path,
                               frames: list[int] | None = None) -> dict[int, np.ndarray]:
    """Load entrant-native deformation gradients from trajectory columns.

    C1/C2 transport attribution requires the exact F used by the entrant's
    covariance update. A position-fitted F remains the kinematic diagnostic,
    but is not accepted as a substitute for native transport state.
    """
    df = _read_trajectory_table(Path(results_dir))
    cols = tuple(f"F{i}{j}" for i in range(3) for j in range(3))
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SubmissionError(
            "C1/C2 transport attribution requires entrant-native deformation "
            f"gradient columns; missing {missing}.")
    out: dict[int, np.ndarray] = {}
    want = set(frames) if frames is not None else set(df["frame"].unique())
    for fr in sorted(want):
        sub = df[df["frame"] == fr].sort_values("particle_id")
        if len(sub) == 0:
            raise SubmissionError(f"trajectory has no native F rows for frame {fr}.")
        out[int(fr)] = sub[list(cols)].to_numpy(dtype=np.float64).reshape(-1, 3, 3)
    return out


def fit_local_deformation_gradients(
        rest: np.ndarray, deformed: np.ndarray, particle_mask: np.ndarray,
        n_neighbors: int = 26, chunk_size: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one local affine deformation gradient per selected particle.

    For particle i, the least-squares problem is

        min_F sum_j ||(x_j-x_i) - F (X_j-X_i)||^2,

    over its nearest rest-state neighbors.  The fit uses positions only and
    is therefore independent of an entrant's internal deformation-gradient
    state.  Returns ``(selected_indices, F)`` with F shaped (M, 3, 3).

    Neighbor search is chunked NumPy rather than a simulator or geometry
    dependency.  C1 and C2 use regular frozen lattices; 26 neighbors cover
    the complete 3x3x3 stencil around an interior particle.
    """
    rest = np.asarray(rest, dtype=np.float64)
    deformed = np.asarray(deformed, dtype=np.float64)
    mask = np.asarray(particle_mask, dtype=bool)
    if rest.shape != deformed.shape or rest.ndim != 2 or rest.shape[1] != 3:
        raise SubmissionError(
            "Local affine fit requires matching rest/deformed arrays of "
            f"shape (N, 3), got {rest.shape} and {deformed.shape}.")
    selected = np.flatnonzero(mask)
    if len(selected) == 0:
        raise SubmissionError("Local affine fit received an empty mask.")
    k = min(int(n_neighbors), len(rest) - 1)
    if k < 6:
        raise SubmissionError(
            "Local affine fit requires at least six neighboring particles.")

    fits = np.empty((len(selected), 3, 3), dtype=np.float64)
    for start in range(0, len(selected), int(chunk_size)):
        stop = min(start + int(chunk_size), len(selected))
        ids = selected[start:stop]
        delta = rest[None, :, :] - rest[ids, None, :]
        dist2 = np.einsum("bij,bij->bi", delta, delta)
        # Include k+1 candidates because the query particle has distance 0,
        # then remove it explicitly.  Stable sorting makes tied lattice
        # neighbors deterministic across platforms.
        candidates = np.argpartition(dist2, kth=k, axis=1)[:, :k + 1]
        for local, pid in enumerate(ids):
            cand = candidates[local]
            cand = cand[cand != pid]
            cand = cand[np.argsort(dist2[local, cand], kind="stable")][:k]
            dX = rest[cand] - rest[pid]
            dx = deformed[cand] - deformed[pid]
            if np.linalg.matrix_rank(dX) < 3:
                raise SubmissionError(
                    f"Local affine neighborhood for particle {pid} is "
                    "rank deficient; cannot estimate a 3D deformation.")
            # dX @ F.T = dx
            fits[start + local] = np.linalg.lstsq(dX, dx, rcond=None)[0].T
    return selected, fits


def load_meta(results_dir: str | Path) -> dict:
    path = Path(results_dir) / "meta.json"
    if not path.exists():
        raise SubmissionError(
            f"No meta.json in {results_dir}. Every submission directory "
            "requires one, see SPEC.md section 2.2.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_meta_declarations(scene: dict,
                            results_dir: str | Path) -> tuple[dict, list[str]]:
    """Generic required-declaration mechanism (SPEC.md section 2.2.1).

    When a scene's pass_criteria contains `requires_meta_declaration`, a
    mapping of dotted meta.json paths to their allowed values (an empty
    list means any value), this checks the submission's meta.json and
    returns (declared, problems):
      declared: {last_path_component: value} for every satisfied path
      problems: human-readable strings for every missing or invalid one

    A nonempty problems list means the submission is INVALID for that
    scene: the scene's result cannot be interpreted without the
    declaration, so no pass/fail judgment is made. Any scene can reuse
    this by adding the pass_criteria key; nothing here is scene-specific.
    """
    req = scene.get("pass_criteria", {}).get("requires_meta_declaration")
    if not req:
        return {}, []
    meta = load_meta(results_dir)
    declared: dict = {}
    problems: list[str] = []
    for path, allowed in req.items():
        node: Any = meta
        for part in path.split("."):
            node = node.get(part) if isinstance(node, dict) else None
            if node is None:
                break
        if node is None:
            problems.append(
                f"meta.json is missing the required declaration '{path}'. "
                f"Declare one of {allowed} (see SPEC.md section 2.2.1); "
                "note that 'notes' must then be a JSON object, with free "
                "text under 'notes.text'.")
        elif allowed and node not in allowed:
            problems.append(
                f"meta.json declaration '{path}' is '{node}', which is not "
                f"in the allowed set {allowed}.")
        else:
            declared[path.split(".")[-1]] = node
    return declared, problems


# ---------------------------------------------------------------------------
# Verdicts and reports
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Verdict:
    """The outcome of one scorer on one submission."""

    scene_id: str
    test_class: str
    tier: str
    status: str                      # PASS / FAIL / NA / INVALID / ERROR
    measured: dict[str, Any] = dataclasses.field(default_factory=dict)
    reference: dict[str, Any] = dataclasses.field(default_factory=dict)
    tolerance: dict[str, Any] = dataclasses.field(default_factory=dict)
    notes: list[str] = dataclasses.field(default_factory=list)
    plots: list[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def verdict_from_scene(scene: dict, status: str, **kw) -> Verdict:
    return Verdict(scene_id=scene["scene_id"], test_class=scene["test_class"],
                   tier=scene["tier"], status=status, **kw)


def write_verdict(verdict: Verdict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(verdict.to_dict(), f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def generate_report(verdicts: list[Verdict], out_dir: str | Path) -> None:
    """Write report.json and report.md. Never a single scalar score."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump([v.to_dict() for v in verdicts], f, indent=2,
                  default=_json_default)

    lines = ["# GaussianBench report", ""]
    for tier in sorted({v.tier for v in verdicts}):
        tier_v = [v for v in verdicts if v.tier == tier]
        n_pass = sum(1 for v in tier_v if v.status == STATUS_PASS)
        n_fail = sum(1 for v in tier_v if v.status == STATUS_FAIL)
        n_na = sum(1 for v in tier_v if v.status == STATUS_NA)
        n_other = len(tier_v) - n_pass - n_fail - n_na
        lines.append(
            f"**Tier {tier}**: {n_pass} pass, {n_fail} fail, {n_na} n/a"
            + (f", {n_other} invalid/error" if n_other else ""))
    lines += ["", "| scene | class | tier | status | measured | reference | tolerance |",
              "|---|---|---|---|---|---|---|"]
    for v in verdicts:
        meas = "; ".join(f"{k}={_fmt(x)}" for k, x in v.measured.items()) or ""
        ref = "; ".join(f"{k}={_fmt(x)}" for k, x in v.reference.items()) or ""
        tol = "; ".join(f"{k}={_fmt(x)}" for k, x in v.tolerance.items()) or ""
        lines.append(f"| {v.scene_id} | {v.test_class} | {v.tier} "
                     f"| **{v.status}** | {meas} | {ref} | {tol} |")
    lines.append("")
    for v in verdicts:
        if v.notes or v.plots:
            lines.append(f"### {v.scene_id}")
            for note in v.notes:
                lines.append(f"- {note}")
            for plot in v.plots:
                lines.append(f"![{Path(plot).stem}]({plot})")
            lines.append("")
    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _fmt(x) -> str:
    if isinstance(x, float):
        if x == 0 or (1e-3 <= abs(x) < 1e4):
            return f"{x:.4g}"
        return f"{x:.3e}"
    return str(x)
