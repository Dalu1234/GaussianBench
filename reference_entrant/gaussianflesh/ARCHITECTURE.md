# GaussianFlesh architecture

GaussianFlesh is included as the full-contract reference entrant. The validated simulator remains in `gaussian_mesh_poc.py` because moving Taichi fields and kernels across modules can change initialization order and compiled-kernel behavior. The public release therefore documents the boundaries without refactoring the verified numerical path.

## Module map

- `gaussian_mesh_poc.py`: validated simulation orchestration, material presets, Taichi fields and kernels, contact, thermodynamics, and interactive visualization.
- `solver_tl.py`: total-Lagrangian solver path.
- `solver_ul.py`: updated-Lagrangian MLS-MPM solver path.
- `gaussian_cli.py`: command-line argument definitions and validation.
- `gaussian_geometry.py`: geometry and particle initialization helpers.
- `gs_render.py`: Gaussian covariance conversion and rasterizer integration.
- `export/`: GaussianBench state export and frozen-scene runners.
- `tests/`: lightweight geometry, CLI, and export-contract tests.

## Stability policy

The large orchestration module is intentionally preserved for the archived reference release. Future development may split rendering, interaction, materials, and thermodynamics into packages, but that refactor should be accompanied by trajectory-level regression tests against this version. This keeps code organization work from silently changing the paper's reported simulation behavior.
