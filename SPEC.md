# GaussianBench Specification

Version: 1.2.0

This document is normative. Words like MUST, SHOULD, and MAY follow RFC 2119 usage.
Anything not specified here is implementation freedom for the submitter.

## 1. Overview

GaussianBench scores a physics-integrated Gaussian splatting simulator against analytical
physics references. The benchmark communicates with your simulator through files only.
You run the scene in your own system, dump a trajectory in the format below, and the
scorers grade the files. No scorer imports simulator code.

Three tiers:

- **Tier A** normally requires particle positions over time (velocities optional but
  rewarded with tighter tolerance). Representation-native A6 instead accepts the
  compact aggregate table specified in Section 2.5.
- **Tier B** also requires only particle positions, but grades relational behavior such
  as per-particle constitutive dispatch and coupling across material interfaces.
- **Tier C** additionally requires per-particle 3x3 covariances over time, rendered
  images, or a representation-native Gaussian parameterization such as C5 triangles.

Every test declares a `test_class` drawn from computational mechanics V&V practice
(ASME V&V 10 tradition):

| test_class | Question it answers |
|---|---|
| `code_verification` | Does the implementation honor its own mathematical contract? |
| `solution_verification` | Does the discrete solution converge to the known continuum solution? |
| `model_validation` | Does the model match physical reality (experimental data)? |
| `capability` | Can the system do this at all, and how well? N/A is allowed. |

Report statuses: `PASS`, `FAIL`, `NA` (system does not support the capability and did not
submit), `INVALID` (submission violated a regime assumption, so the reference does not
apply and no pass/fail judgment is possible), `ERROR` (submission was malformed).

`NA` and `FAIL` are different outcomes and MUST never be conflated in reports.

## 2. Submission layout

A results submission is a directory per scene:

```
results/<your_system_name>/<scene_id>/
  trajectory.parquet        (preferred)  OR  trajectory.csv
  meta.json                 (required)
  covariances.parquet       (Tier C scenes only)
  renders/                  (C3 only)
    before.png
    after.png
    camera.json
```

### 2.1 trajectory.parquet / trajectory.csv

One row per (frame, particle). Columns:

| column | type | unit | required |
|---|---|---|---|
| `frame` | int | index from 0 | yes |
| `time` | float | seconds | yes |
| `particle_id` | int | stable across frames | yes |
| `x`, `y`, `z` | float | meters | yes |
| `vx`, `vy`, `vz` | float | m/s | no |
| `mass` | float | kg | no |
| `phase` | int, 0 solid 1 fluid | dimensionless | no (required for A4) |

Rules:

- `particle_id` MUST be stable: the same physical particle keeps the same id in every frame.
- Every frame MUST contain the same set of particle ids (no birth/death within a run).
- `time` MUST be strictly increasing with `frame` and identical for all particles in a frame.
- Frame 0 MUST be the initial condition exactly as given by the scene positions file,
  in the same order (particle_id i corresponds to row i of the scene `.npy`).
- Output frame rate MUST match `simulation.output_fps` in the scene JSON within 1%.
  Internal substepping is your business; only output sampling is specified.
- Positions are in scene coordinates (meters). Do not rescale, recenter, or rotate.

If velocity columns are absent, scorers that need velocities (A1) derive them by central
finite differences of positions and apply the looser `derived_velocity` tolerance tier
declared in the scene `pass_criteria`. The report flags that derived velocities were used.

If `mass` is absent, scorers assume equal mass for every particle, with total mass
inferred from the scene material density times the scene geometry volume. This is stated
in the report as an assumption.

`phase` is required for A4 (melt front). The melt front cannot be inferred from positions
alone, because a molten and a solid particle at the same location are indistinguishable
without state. Systems that cannot output phase MUST NOT submit A4 and are marked NA.

### 2.2 meta.json

```json
{
  "system_name": "GaussianFlesh",
  "system_version": "0.9",
  "spec_version": "1.0.0",
  "columns_present": ["vx", "vy", "vz", "mass"],
  "wall_clock_seconds": 214.5,
  "hardware": "RTX 4070 Laptop, i9-13900H",
  "notes": "Lateral boundaries are traction-free. Grid BC only at floor."
}
```

`notes` is free text for declared deviations from the scene spec. Declaring a deviation
does not excuse it, but an undeclared deviation discovered later invalidates the entry.

Since spec 1.1.0, `notes` MAY instead be a JSON object. In that case free text goes
under `notes.text`, and the remaining keys are structured declarations that scenes can
require (see 2.2.1). The string form remains valid for scenes that require no
declarations.

### 2.2.1 Required declarations (`requires_meta_declaration`)

A scene MAY carry, inside its `pass_criteria`, a mapping from dotted meta.json paths to
allowed values:

```json
"requires_meta_declaration": {
  "notes.domain_handling": ["fixed", "co_moving", "sparse", "mesh_free", "other"]
}
```

The scorer resolves each path inside the submission's meta.json. A missing path, or a
value outside the allowed set (an empty set allows any value), makes the submission
INVALID for that scene, not FAIL: the result cannot be interpreted without the
declaration, so no pass/fail judgment is possible. Satisfied declarations are echoed
into the verdict's measured block (as `declared_<key>`) so reports can group results by
category. The mechanism is generic; any scene can use it.

The first user is `a1b_conservation_drifting_v1`, which requires
`notes.domain_handling`: how the submitting system's computational domain follows (or
does not follow) a body that translates far from its starting point. A `fixed` entrant's
conservation numbers mean something different from a `co_moving` entrant's, and the
report must be able to say which is which.

### 2.3 covariances.parquet (Tier C)

One row per (frame, particle). Columns: `frame`, `particle_id`, and the six independent
entries of the symmetric world-space covariance in meters squared:
`sxx, sxy, sxz, syy, syz, szz`, so that

```
Sigma = [[sxx, sxy, sxz],
         [sxy, syy, syz],
         [sxz, syz, szz]]
```

Frame 0 covariances count as the rest covariances. Scorers reconstruct predicted
deformed covariances from frame 0, so frame 0 MUST reflect the undeformed state.

C1 and C2 additionally require the nine entrant-native deformation-gradient
columns `F00` through `F22` in `trajectory.parquet`. These must be the exact
per-particle gradients used by the submitted covariance update. Position-fitted
gradients are computed independently for the kinematic-realization diagnostic
and are never substituted into the pure transport reference.

C2 requires full per-frame covariance coverage at output_fps, including the scene-declared
peak-shear frame. Its scorer first grades the exact peak $F\Sigma_0F^\top$ transport and then
grades full-trajectory covariance health. C1 only strictly needs frame 0 and the final frame
(submitting all frames is fine, scorers select what they need).

### 2.4 renders/ (C3 only)

- `before.png`: rendered at frame 0 from the camera in `camera.json`.
- `after.png`: rendered at the final frame from the identical camera.
- `camera.json`: `{ "position": [x,y,z], "look_at": [x,y,z], "up": [x,y,z], "fov_deg": f, "width": w, "height": h }`.

Both images MUST be the same resolution, rendered by the submitter's own renderer.
This is deliberate: the contract under test is your physics-to-appearance link, not
our renderer. Any tone mapping or background must be identical between the two images.

### 2.5 body_metrics.csv (A6 only)

A6 accepts a compact representation-native table with one row per frame and
columns `frame`, `time`, `com_x`, `com_y`, `com_z`, `com_vx`, `com_vy`,
`com_vz`, `bbox_volume`, `detF_nonpositive_fraction`, `nonfinite_fraction`,
and `particle_count`. The released Physics3D HDF5 adapter computes these
quantities directly from native particle states. Frame indices MUST be
contiguous from zero, times MUST increase from zero, and particle count MUST
remain fixed.

### 2.6 triangle_metrics.csv (C5, C6, C7, C8)

The representation-native shape scenes accept a compact stable-triangle table
with one row per frame and columns `frame`, center of mass,
`edge_rel_error_median`, `edge_rel_error_p95`, `area_rel_error_median`,
`area_rel_error_p95`, `nonfinite_fraction`, and `triangle_count`. Errors are
evaluated against frame-zero triangle identity. The released GASP adapter
computes the table from native `.pt` triangle-soup checkpoints. Triangle
ordering and count MUST remain stable.

C5 grades transport fidelity through substantial motion. C6 grades Galilean
rigid transport (uniform velocity, zero gravity: straight constant-speed
center-of-mass path plus undisturbed shape). C7 grades long-horizon rest
stability on the system's own released static support and additionally
requires the `bbox_volume` column, used solely as the body's own length scale
for its drift bound. C8 grades robustness and elastic recovery through a hard
floor impact, with a deformation gate so a too-gentle run is INVALID rather
than a cheap PASS. Systems whose native stable-identity structure is not a
triangle soup MAY compute the same columns over their own persistent element
structure (for example spring rest lengths) and MUST declare that mapping in
`meta.json` notes.

### 2.7 stress_strain.csv / stress_strain.parquet (M1 only)

M1 is a material-model validation rather than a trajectory test. Its scene directory
contains `meta.json` plus exactly one stress-strain table with these columns:

| column | type | unit | required |
|---|---|---|---|
| `stretch` | float | ratio | yes |
| `cauchy_stress_pa` | float | Pa | yes |

Samples MUST describe a quasi-static, isothermal uniaxial-tension response with
traction-free transverse faces. At least five finite samples must fall in
`1.001 < stretch <= 2.5`. A system that cannot expose constitutive stress MUST omit M1
and is reported as NA; it MUST NOT substitute a structural trajectory.

## 3. Scene format

Each scene is a single self-describing JSON file in `scenes/`. Example:

```json
{
  "scene_id": "a2_beam_frequency_v1",
  "test_class": "solution_verification",
  "tier": "A",
  "description": "Slender cantilever pluck, fundamental frequency vs Euler-Bernoulli.",
  "geometry": {
    "type": "box",
    "size_m": [0.4, 0.05, 0.05],
    "particle_spacing_m": 0.005,
    "particle_positions_file": "a2_beam_positions.npy"
  },
  "material": {
    "model_hint": "linear_elastic_or_equivalent",
    "youngs_modulus_pa": 5.0e5,
    "poisson_ratio": 0.3,
    "density_kg_m3": 1000
  },
  "boundary_conditions": { "clamp": { "region": "x < 0.01", "type": "fixed" } },
  "initial_conditions": { "tip_deflection_m": 0.02, "release_at_t": 0.0 },
  "simulation": { "duration_s": 4.0, "output_fps": 240, "gravity": [0, 0, 0] },
  "pass_criteria": {
    "quantity": "fundamental_frequency_hz",
    "reference": "euler_bernoulli_cantilever",
    "rel_tolerance": 0.10,
    "damping_report_only": true
  },
  "provenance": {
    "reference_formula": "f1 = (1.875104^2 / (2*pi*L^2)) * sqrt(E*I / (rho*A))",
    "regime_assumptions": ["small deflection", "slender beam", "linear elastic"]
  }
}
```

Normative points:

- For standard trajectory scenes, `particle_positions_file` names an `.npy` file
  (float64, shape (N, 3), meters) shipped in `scenes/`. Every entrant MUST start
  from literally these positions. Row index is the `particle_id`. A6 and C5 are
  explicitly representation-native and use the frozen contracts in Sections 2.5
  and 2.6 instead.
- `pass_criteria` is frozen once a scene version ships. Changing a tolerance bumps the
  scene version suffix (`_v1` to `_v2`) and is a recorded event in the changelog.
- `pass_criteria` carries every tolerance the scorer uses, including the looser
  `derived_velocity` tiers where relevant. Scorers contain no hardcoded tolerances.
- `provenance` states the reference formula and its regime assumptions. If a submission
  leaves the regime (for example, deflection too large for the small-deflection formula),
  the scorer emits INVALID, not FAIL.
- `material.model_hint` tells the submitter what constitutive behavior the reference
  assumes. Your system maps this onto its own material model and declares the mapping
  in `meta.json` notes.

## 4. How scoring works

```
python run_all.py --results-dir results/<your_system_name> --scenes-dir scenes/
```

For every scene JSON, the runner checks whether the results directory contains the
required files for that scene. Missing optional-capability files mean NA. Present but
malformed files mean ERROR. Otherwise the scorer runs and emits PASS, FAIL, or INVALID
with measured values, reference values, and tolerances, into `report.json` and a
human-readable `report.md` in the results directory.

The report is a per-test vector. There is no single aggregate score, by design: a single
number invites gaming and hides which capability failed.

## 5. Versioning

- Spec version: this file, semver. Breaking format changes bump major.
- Scene version: suffix on `scene_id`. Tolerances are frozen per version.
- Superseded scenes move to `scenes/deprecated/` with their positions files and frozen
  criteria intact, so historical results stay reproducible; they are not discovered by
  the runner.
- Held-out variants: `variants/generate.py` produces randomized scenes from a seed.
  Public seeds are committed for practice; evaluation seeds are not.

### 5.1 The A1 split (v0.2.0)

The original `a1_conservation_v1` (spinning ball drifting 11.5 m over 10 s) conflated
two separate diagnostics: whether the particle-grid transfer conserves momentum, and
whether the system's domain handling can keep a small body resolved while it translates
far across the computational domain. A fixed-grid method with a perfectly conserving
transfer could fail it purely because covering the drift envelope forced its grid so
coarse the body became sub-grid. That is a real limitation worth measuring, but it is
not the code-verification claim A1 was written to test, so the scene was split:
`a1a_conservation_inplace_v1` removes the translation and keeps the original tight
tolerances (any violation is attributable to the transfer scheme), while
`a1b_conservation_drifting_v1` keeps the original physics with tolerances that reflect
its compound nature and a required `domain_handling` declaration. The relaxation in A1b
is a re-scoping, not an easing: the tight test still exists, in A1a.
