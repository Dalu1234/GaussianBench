# Contributing

Contributions are welcome when they preserve the benchmark's frozen-contract design.

## Before opening a pull request

1. Install the scorer environment using `INSTALL.md`.
2. Run `python -m pytest -q` from the repository root.
3. Run `python make_example.py` followed by `python run_all.py --results-dir results/example`.
4. Run `python adapters/check_release_artifact.py` when changing adapters, provenance, or packaged reports.
5. Do not change a released scene, reference, validity gate, or tolerance in place. Add a new versioned scene instead.
6. Keep simulator-specific physics inside the released system. Adapters may map units, coordinates, inputs, and exported state, but may not replace the system's solver.
7. Record upstream commit hashes and all compatibility patches for external entries.

Please keep generated trajectories, videos, caches, checkpoints, and local absolute paths out of commits. New scorers require synthetic PASS, FAIL, INVALID, and NA tests where applicable.
