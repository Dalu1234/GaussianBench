# GaussianFlesh Representation

Research prototype for particle-based Gaussian deformation and rendering. The main simulator is `gaussian_mesh_poc.py`, with reusable CLI parsing in `gaussian_cli.py`, geometry helpers in `gaussian_geometry.py`, and focused reference-comparison scripts around the PhysGaussian ficus scene. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for module boundaries and the reason the validated Taichi orchestration path remains intact in this release.

## Known Environment

This branch has been tested on:

- Windows 11
- Python 3.12
- NVIDIA CUDA GPU via Taichi CUDA
- NVIDIA GeForce RTX 4070 Laptop GPU
- Taichi 1.7.4
- Warp `warp-lang==1.8.1` for PhysGaussian compatibility

The primary runtime expects CUDA. CPU execution is not the validated path for the current public cleanup branch. The lightweight unit tests do not require a full simulation run.

## Setup

Create and activate a virtual environment. For the full reference implementation:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For only the lightweight export/unit tests, the smaller dev requirements are:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

The PhysGaussian reference-comparison harness requires the upstream PhysGaussian checkout as a sibling directory by default:

```text
C:\Users\<you>\Desktop\Programming\GaussianFlesh
C:\Users\<you>\Desktop\Programming\PhysGaussian
```

You can override that location with `PHYSGAUSSIAN_ROOT` or `--phys-root` when using `physgaussian_ficus_reference.py`.

## Tests

Run the lightweight Python test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Current cleanup-branch verification: `12 passed`.

## Normal GaussianFlesh Demos

The general simulator entry point is `gaussian_mesh_poc.py`.

Rubber ficus visual demo:

```powershell
.\.venv\Scripts\python.exe gaussian_mesh_poc.py `
  --ply C:\Users\<you>\Desktop\Programming\PhysGaussian\model\ficus_whitebg-trained\point_cloud\iteration_30000\point_cloud.ply `
  --ply-upright `
  --material 1 `
  --ply-split 1:0.01 `
  --ply-split-axis y `
  --impact-vy -5.5 `
  --frames 360 `
  --video poster_assets\ficus_rubber_bounce_public_cleanup.mp4 `
  --gs-only
```

This is a visual rubber bounce demo. It is not the PhysGaussian reference-comparison setup.

## PhysGaussian Reference Comparison

GaussianFlesh is not a copy of PhysGaussian. This comparison keeps one reproduced PhysGaussian ficus setup as a sanity check for the UL-MLS-MPM path before testing GaussianFlesh-specific features such as material splits, thermodynamics, and custom visual demos.

The comparison has two sides:

1. PhysGaussian reference state/video from `physgaussian_ficus_reference.py`.
2. GaussianFlesh matched state/video from `gaussian_mesh_poc.py --physgaussian-ficus-match` or `--physgaussian-ficus-drop`.

Generate the PhysGaussian reference states:

```powershell
.\.venv\Scripts\python.exe physgaussian_ficus_reference.py phys `
  --output comparison_runs\phys_10f `
  --frames 10
```

Generate the GaussianFlesh states under the matched reference settings:

```powershell
.\.venv\Scripts\python.exe gaussian_mesh_poc.py `
  --ply C:\Users\<you>\Desktop\Programming\PhysGaussian\model\ficus_whitebg-trained\point_cloud\iteration_30000\point_cloud.ply `
  --solver ul_mlsmpm `
  --physgaussian-ficus-match `
  --state-output comparison_runs\ours_10f `
  --frames 10
```

Compare them:

```powershell
.\.venv\Scripts\python.exe physgaussian_ficus_reference.py compare `
  --phys comparison_runs\phys_10f `
  --ours comparison_runs\ours_10f `
  --output comparison_runs\comparison_10f.json
```

Recent cleanup-branch frame-10 comparison:

```text
position mean error: 2.060456e-05
velocity mean error: 4.450028e-05
F mean error:        7.339840e-05
```

Generate matched reference-comparison videos:

```powershell
.\.venv\Scripts\python.exe physgaussian_ficus_reference.py video `
  --output poster_assets\physgaussian_ficus_baseline_public_cleanup.mp4 `
  --frames 125

.\.venv\Scripts\python.exe gaussian_mesh_poc.py `
  --ply C:\Users\<you>\Desktop\Programming\PhysGaussian\model\ficus_whitebg-trained\point_cloud\iteration_30000\point_cloud.ply `
  --solver ul_mlsmpm `
  --physgaussian-ficus-drop `
  --video poster_assets\gaussianflesh_ficus_baseline_public_cleanup.mp4 `
  --frames 125
```

Use `warp-lang==1.8.1` for this path. Newer Warp releases reject older PhysGaussian kernel syntax.

## Thermodynamics Demo

Wax-melt ficus render from the clean branch:

```powershell
.\.venv\Scripts\python.exe gaussian_mesh_poc.py `
  --ply C:\Users\<you>\Desktop\Programming\PhysGaussian\model\ficus_whitebg-trained\point_cloud\iteration_30000\point_cloud.ply `
  --ply-upright `
  --material w `
  --frames 600 `
  --video poster_assets\ficus_wax_melt_public_cleanup.mp4 `
  --gs-only `
  --thermo `
  --hot-floor 200
```

`--material w` selects the wax material. `--thermo` enables heat diffusion, latent heat, thermal softening, and solid-to-fluid phase transition.

## Generated Assets

Videos, captures, benchmark outputs, comparison outputs, logs, and local caches are generated artifacts. They are intentionally ignored by `.gitignore`, including:

- `poster_assets/`
- `comparison_runs/`
- `captures/`
- `results/`
- `bench_results*/`
- `*.mp4`
- `*.log`

Regenerate them locally with the commands above instead of committing them to source control.
