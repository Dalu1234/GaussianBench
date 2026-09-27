# Installation

GaussianBench and the GaussianFlesh reference entrant have separate environments. The benchmark scorers are CPU-only. The reference entrant additionally requires a CUDA-capable NVIDIA GPU for its validated simulation path.

## Benchmark scorers

Tested with Python 3.11 and 3.12.

```bash
python -m venv .venv
```

Activate the environment, then install and verify the benchmark:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
python make_example.py
python run_all.py --results-dir results/example
python adapters/check_release_artifact.py
```

Expected results are 66 passing scorer tests, 9 PASS and 9 NA for the synthetic example, and `AUDIT PASSED` from the release audit.

## GaussianFlesh reference entrant

Create a second environment so GPU and rendering dependencies do not affect the benchmark scorer environment:

```bash
cd reference_entrant/gaussianflesh
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

The lightweight reference-entrant tests should report 12 passes. Full simulations use Taichi 1.7.4 with CUDA. PhysGaussian comparison commands additionally require a separately downloaded PhysGaussian checkout. See `reference_entrant/gaussianflesh/README.md` for the tested configuration and commands.

## External entries

Third-party systems are not vendored. Their exact revisions, compatibility patches, setup commands, state mappings, and result directories are documented in `REPRODUCING_EXTERNAL_ENTRIES.md`.
