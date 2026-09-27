# How to reproduce the external entries

This guide explains how we connected each released system to GaussianBench and
how to check that the connection is fair. The basic workflow is simple: each
system runs in its own environment, writes its native state to disk, and is then
scored by GaussianBench. The scorer never imports the simulator.

There are two useful levels of reproduction. If you want to check the benchmark
and its file contract, the first section takes a few minutes and does not need a
GPU. If you want to inspect or rerun a published system, the remaining sections
record the exact revision, runner, patch, and state mapping used for that entry.

## 1. Check the benchmark first

From the GaussianBench directory, run:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m pytest -q
python make_example.py
python run_all.py --results-dir results/example
```

The tests should report `66 passed`. The example report should contain 9 PASS
and 9 NA, with no ERROR. It uses closed-form synthetic trajectories to exercise
the file format and scorers; it is not one of the paper's evaluated systems.

The paper's lightweight reports are in `reports/` at the root of the
supplementary package. Each row includes the measured value, expected value,
tolerance, verdict, and notes. We leave out the multi-gigabyte trajectories,
but include the runner and patch that generated them.

You can also check the packaged reports directly:

```bash
python adapters/check_release_artifact.py
```

This command checks the release pins, required metadata, verdict labels, and
local-path cleanup. It only reads the reports; it does not rescore or modify
them.

## 2. What the adapter does

The adapter layer is intentionally small. It reads a frozen scene, places that
scene inside the released solver's coordinate system, passes the requested
material and boundary parameters through the method's native interface, and
exports the resulting state in GaussianBench's Parquet or CSV format. If a
coordinate transform is needed, the exact inverse is applied on export.

We do not use the adapter to replace a solver or invent state that the method
does not expose. We also do not change a scene tolerance for a particular
method. If a system has no native version of a required state, that scene is
NA. If it attempts a scene but does not realize a required boundary motion or
validity condition, the result is INVALID rather than a physics FAIL. This
keeps adapter limitations separate from solver failures.

The formal checklist is in `adapters/VALIDATION_PROTOCOL.md`. The rest of this
guide explains how it was applied in practice.

## 3. Revisions and reproduction files

| System | Upstream revision | Runner included here | Patch included here | Report |
|---|---|---|---|---|
| PhysGaussian | `8339ed6aa2cd5d50e1001a254a3d95aea678a956` | `adapters/reproduction/PhysGaussian_shared/run_bench_scene.py` | none | `PhysGaussian_native_2026-08-27` |
| GaussianFluent | `de7e10675b9dce58952241087bb15cce9bc3f6ef` | shared Warp-MPM runner and `adapters/gaussianfluent_to_bench.py` | `adapters/reproduction/GaussianFluent/headless.patch` | `GaussianFluent_full` |
| OmniPhysGS | `5ab915b014467b911581b0c6df351713d3114a08` | `adapters/reproduction/OmniPhysGS/run_gaussianbench.py` | `adapters/reproduction/OmniPhysGS/compatibility.patch` | `OmniPhysGS_full` |
| PhysDreamer | `95beb71f9f1b037b05d7c364d1943dde10b0efbd` | `adapters/reproduction/PhysDreamer/run_gaussianbench.py` | `adapters/reproduction/PhysDreamer/compatibility.patch` | `PhysDreamer_full` |
| Physics3D | `32d92183e2918f26384ec60e410a8145d6efafd9` | shared runner and `adapters/reproduction/Physics3D/run_gaussianbench.py` | `adapters/reproduction/Physics3D/reproduction.patch` | `Physics3D_broad` |
| GASP | `186c9c912df8b4a4258810e6e4754e8cfcf0c53d` | `adapters/reproduction/GASP/run_gaussianbench.py` | none | `GASP_broad` |
| Spring-Gaus | `62a1bb5dbe83fe4396efa7048d3226754ac8fe1d` | `adapters/reproduction/Spring-Gaus/run_gaussianbench.py` | `adapters/reproduction/Spring-Gaus/compatibility.patch` | `Spring-Gaus_full` |
| PhysTwin | `81c718790a37e5e0102eb77af2c6edd34a9db25f` | `adapters/reproduction/PhysTwin/run_gaussianbench.py` | `adapters/reproduction/PhysTwin/compatibility.patch` | `PhysTwin_full` |

Most of the patches are mundane compatibility fixes: making visualization
optional, replacing a removed library call, or avoiding an import that is not
needed for simulation. They do not change forces, stresses, or integration.
Physics3D required a more substantial reproduction patch, so we discuss it
separately below and include the full diff for inspection.

Report metadata identifies the clean upstream revision in `system_version` and
records any benchmark-time modification separately as `compatibility_patch`
and `compatibility_patch_sha256`. This avoids treating a patched working tree as
an unnamed revision while preserving the exact source delta used for the run.
The packaged patch hashes are:

- GaussianFluent `headless.patch`: `b91788de7b115326a3722ef5e5ea36859b0b4decad8976168ebbfc67669b5e51`
- Physics3D `reproduction.patch`: `bcf9163211d00424628d86cbf7389399382939166510a7fe3cb329e0f8dda42a`

For clarity, `8339ed6a...` is the released PhysGaussian revision returned by
`git rev-parse HEAD` and is present on the upstream main branch. The shared
runner was added as an untracked benchmark harness on top of that checkout, so
it does not honestly have a separate Git commit. We therefore ship the exact
runner files and their hashes in
`adapters/reproduction/PhysGaussian_shared/MANIFEST.sha256`. Reports record the
method revision and, where applicable, the separate harness base revision.

We also reran all seven GaussianFluent scenes while preparing this artifact,
with its method root and system name set explicitly. Every verdict matched the
earlier profile. The verified C1 run records GaussianFluent revision
`de7e1067...` and the separate shared-harness revision `8339ed6a...`. In the
matched covariance fault, the kinematic median stays fixed at `0.8643%`, while
transport changes from `0.0112%` (PASS) to `51.51%` (FAIL). The report included
in this package is that provenance-verified rerun.

## 4. Setting up the released methods

A sibling-directory layout is the easiest to work with:

```text
workspace/
  gaussianbench/
  PhysGaussian/
  GaussianFluent/
  omniphysgs/
  PhysDreamer/
  Physics3D/
  GASP/
  Spring-Gaus/
  PhysTwin/
```

For a method you want to rerun, check out the revision in the table, apply its
included patch if it has one, and copy the included runner into that method's
`bench/` directory. Each method should have its own environment because the
released projects use different Python, CUDA, PyTorch, Warp, and Taichi
versions. GaussianBench's scorer environment should remain separate.

Most runners use the same command shape:

```bash
python bench/run_gaussianbench.py \
  --scene ../gaussianbench/scenes/<scene_id>.json \
  --out ../gaussianbench/results/<system>/<scene_id>

python ../gaussianbench/run_all.py \
  --results-dir ../gaussianbench/results/<system>
```

PhysGaussian and the shared Physics3D path use:

```bash
python bench/run_bench_scene.py \
  --scene ../gaussianbench/scenes/<scene_id>.json \
  --out ../gaussianbench/results/<system>/<scene_id> \
  --tier-c
```

If GASP is not beside GaussianBench, point `GAUSSIANBENCH_ROOT` to the benchmark
directory. Its controlled faults are selected with `--fault half_gravity`,
`--fault gravity_left_on`, or `--fault sand_material`.

## 5. How each system is mapped

### PhysGaussian and GaussianFluent

These systems use the shared Warp-MPM runner. Frozen positions are translated
into the released grid without changing their scale. The runner sends the scene
values for `E`, `nu`, density, gravity, and time step through the released
parameter path, then removes the translation when exporting results.
Positions, velocities, and `F` are read from native solver arrays. Covariance
comes from the same `compute_cov_from_F` push-forward used before rendering,
with the scene's rest covariance.

The shared runner contains no physics solver of its own. It imports the
released `MPM_Simulator_WARP` from `GAUSSIANBENCH_METHOD_ROOT`; its job is to
configure the frozen scene and serialize native arrays for scoring.

The matched fault freezes only covariance at its frame-zero value. Positions
and `F` are copied unchanged, so any verdict change is attributable to the
transport channel.

GaussianFluent's patch only makes rendering dependencies optional and removes a
machine-specific garden-asset path from headless simulation. Its converter does
not manufacture missing covariance. Without a native covariance export, the
corresponding Tier-C scene remains unentered.

### OmniPhysGS

OmniPhysGS is translated into its released fixed grid and uses its released
per-particle constitutive dispatch. The runner exports native position,
velocity, and `F`, and evaluates covariance using the push-forward used by the
released renderer. Its patch removes unused configuration imports and replaces
an `einops` transpose with the equivalent PyTorch transpose. No physics update
is changed.

### PhysDreamer

PhysDreamer runs through its released differentiable Warp-MPM core. We export
native position, velocity, `F`, and its first-order covariance state. The patch
adds a type-annotation fallback and replaces a removed low-level Warp tensor
conversion with the maintained public call. The solver equations are left
alone. Its covariance fault again changes only the covariance export.

### Physics3D

Physics3D needs a little more explanation because two release details are easy
to miss. First, its `compute_mu_lam_from_E_nu` function expects Young's modulus
in units of `10^7 Pa`. We therefore pass a scene value in pascals as `E/10^7`
and record both values in metadata. Passing raw pascals makes the solver roughly
seven orders of magnitude too stiff, breaks its CFL condition, and produces the
CUDA out-of-bounds failure we initially observed.

Second, GaussianBench's corotated material is matched to the base corotated
stress already present in Physics3D, without the learned Maxwell residual. The
included patch makes that branch selectable as `jelly_base`, initializes the
persistent Maxwell deformation field to identity like its trial field, and adds
a simulation-only export path. Because this patch touches released source, we
include the complete diff rather than asking the reader to take the mapping on
trust.

C1 covariance is read through Physics3D's own renderer export. We check the two
main channels separately: half gravity must be detected by A6, while frozen
covariance must change C1 transport without changing its kinematics.

### GASP

GASP is evaluated through the representation it actually releases: stable
triangles. The runner uses the released Taichi MPM engine, teddy pseudomesh,
`Rescale` transform, per-frame triangle size-control rule, and inverse export.
We do not create particle covariance for it. A6 reads native center of mass and
`F`; C5 through C8 use persistent triangle edges and areas.

Its three matched faults each change one released input: gravity magnitude for
A6, zero-gravity handling for C6, and elastic versus sand material for C8. Each
fault is expected to affect its intended metric rather than producing a generic
failure everywhere.

### Spring-Gaus and PhysTwin

Spring-Gaus and PhysTwin are included as targeted profiles. Their released
persistent points and velocities support the scenes we enter, but neither
release provides the mapping needed to turn GaussianBench's continuum `E,nu`
values into its learned spring parameters. We also do not invent triangle faces
for either method. Those unsupported cells remain NA and are not counted as
additional coverage.

## 6. Following one result from scene to verdict

For any result you want to inspect, these four files tell the complete story:

1. `scenes/<scene_id>.json` states the requested setup, expected behavior,
   validity conditions, and tolerance.
2. `adapters/reproduction/<system>/` contains the runner and any source patch.
3. `reports/<profile>/<scene_id>/meta.json` records the method revision,
   hardware, exported fields, transforms, units, substeps, and deviations.
4. `reports/<profile>/report.json` contains the measured value and verdict.

Different GPUs may produce slightly different floating-point measurements, so
we expect a rerun to reproduce the verdict and approximately reproduce the
continuous metric, not to be bit-for-bit identical. The classification rules do
not change on rerun: a missed realization gate is INVALID, and a state channel
that the method does not provide is NA.
