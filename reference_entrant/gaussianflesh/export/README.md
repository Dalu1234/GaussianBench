# GaussianBench exporter

Runs a GaussianBench scene in GaussianFlesh and writes a submission
directory in the benchmark's SPEC.md format. The two repos stay decoupled:
this module never imports the benchmark package, and the benchmark never
imports the simulator. The only interface is files.

## Run one scene

From the GaussianFlesh repo root, using the repo venv:

```
.\.venv\Scripts\python.exe -m export.run_bench_scene ^
    --scene C:\path\to\gaussianbench\scenes\a1a_conservation_inplace_v1.json ^
    --out bench_results
```

Options: `--tier-c` (record per-frame covariances), `--max-seconds S`
(truncated dry run, declared in meta notes), `--substep-dt DT` (override
the CFL-derived substep).

## Run everything

```
.\.venv\Scripts\python.exe -m export.run_all_scenes
```

One subprocess per scene (mandatory: the simulator fixes particle count
and Taichi field shapes at import time). Then score:

```
python C:\path\to\gaussianbench\run_all.py --results-dir bench_results
```

## What the harness does

- Loads the scene's frozen `.npy` positions verbatim (particle_id = row
  index). Placement transform: sim(t) = P (scene(t) - v_frame t) + offset,
  with the exact inverse applied on export. The optional Galilean boost
  (v_frame) engages for large-drift floorless scenes so the body stays
  near the coordinate origin: physics is Galilean-invariant, and f32
  position rounding grows with |x| (measured: rotation scrambling at
  11.5 m). Declared as domain_handling = "co_moving".
- Converts E, nu to Lame parameters (the one permitted conversion), sets
  per-particle mu, lambda, mass, and rest volume, seeds the APIC affine
  matrix with the IC velocity gradient (C = skew(omega) for rigid spin:
  C = 0 loses (dx/R)^2 of angular momentum to the first transfer).
- Drives `solver_ul.substep(..., use_floor=False)` directly.
- Boundary conditions are enforced at BOTH levels, which MPM requires:
  particle pins hold the prescribed positions exactly, and grid-slab
  velocity overrides (via the substep's grid_bc_fn hook) transmit the
  constraint to the interior. A particle-only pin is mass-diluted by the
  B-spline stencil and transmits about 2 percent of the commanded strain.
  All scene BCs reduce to affine velocity fields v = rate (A x + b) over
  moving slabs.
- meta.json records: git commit, hardware, wall clock, solver path,
  frame/substep dt, Taichi arch and precision (cuda, f32), seeds, the
  placement transform, the material conversion, and every deviation.

## Scene support (verdicts of 2026-07-04, benchmark v0.2.1)

| scene | verdict |
|---|---|
| a1a_conservation_inplace_v1 | FAIL on linear momentum only (1.6e-5 vs the f64-class 1e-6 tier, the f32 floor, FIX_LEDGER entry 3); angular + energy within tolerance |
| a1b_conservation_drifting_v1 | PASS (Galilean-boosted co-moving frame) |
| a2_beam_frequency_v1 | PASS |
| a3_wavespeed_v1 | PASS |
| a4_meltfront_v1 | PASS (front prefactor within 0.06 percent of the Neumann solution; SI-calibrated batched heat with roundtrip-smoothing compensation) |
| a5_stiffness_v1 | PASS |
| c1_covariance_v1 | PASS |
| c2_blobhealth_shear_v1, c2_blobhealth_melt_v1 | PASS |
| c3_roundtrip_v1 | PASS (PSNR 48.2 dB, SSIM 0.9994; unilateral plate + floor planes, gs_render round trip) |
| c4_captured_frequency_v1 | PASS |

10 of 11 scenes pass; the single failing metric is a documented
architectural precision limit, not a physics defect.

## Simulator changes made for the benchmark (all additive)

- `solver_ul.substep(grid_bc_fn=...)`: optional grid-BC callback slot.
- `solver_ul.set_grid_origin()`: runtime-movable world-grid origin.
- Partition-of-unity + first-moment stencil normalization in
  `_ti_build_world_grid_ul`: enforces sum w = 1, sum grad w = 0, and
  sum w dpos = 0 per particle, killing frozen f32 rounding biases that
  integrated into constant spurious forces (measured 240x reduction in
  momentum drift). Benefits every UL scene, not just the benchmark.

