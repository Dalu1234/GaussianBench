# GaussianBench

A physics-fidelity benchmark for physics-integrated Gaussian splatting
simulators. Any system that moves 3D Gaussians with physics (MPM, spring-mass,
position-based dynamics, neural) can be scored against analytical references:
you run the frozen scenes in your own simulator, dump particle trajectories to
files, and the scorers grade the files. No scorer ever imports simulator code.

GaussianBench measures a different question from appearance-only evaluation.
A rendered rollout can resemble a reference video while violating conservation,
wave speed, covariance transport, or phase-boundary dynamics. The suite grades
those claims separately and makes unsupported capabilities (`NA`) and invalid
comparison regimes (`INVALID`) explicit instead of folding them into one visual
score.

This repository contains the benchmark, frozen public reports, exact external
adapter snapshots, and the GaussianFlesh full-contract reference entrant.
See [`INSTALL.md`](INSTALL.md) for environment setup and
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the frozen-scene contribution policy.

- **Tier A** needs only particle positions over time. Any black box can enter.
- **Tier B** adds relational tests such as constitutive dispatch and material-interface coupling, while still requiring only particle positions.
- **Tier C** additionally grades the Gaussian state itself: covariance
  push-forward correctness, blob health under violent deformation, and
  appearance round trips through your own renderer.

The result is a per-test vector, never a single score: a single number invites
gaming and hides which capability failed. `NA` (capability not supported) is a
first-class outcome distinct from `FAIL`, and `INVALID` (submission left the
regime where the analytical reference applies) is distinct from both.

Paired counterfactual protocols are specified in
[`INTERVENTIONS.md`](INTERVENTIONS.md). They hold geometry fixed, alter one
thermal boundary or material painting, and grade the trajectory change against
Neumann/Stefan or series-compliance predictions.

## External systems and reviewer reproduction

The repository includes the exact runner snapshots and compatibility patches used
for every reported external entry. Start with
[`REPRODUCING_EXTERNAL_ENTRIES.md`](REPRODUCING_EXTERNAL_ENTRIES.md), then use
[`adapters/VALIDATION_PROTOCOL.md`](adapters/VALIDATION_PROTOCOL.md) to audit
release pins, units, coordinate transforms, scene realization, native-state
provenance, and matched negative controls. Third-party repositories and model
checkpoints are not redistributed; their frozen revisions are recorded in the
reproduction guide and report metadata.

## Five-minute quickstart

Requirements: Python 3.11 or 3.12. Full installation details are in
[`INSTALL.md`](INSTALL.md). For the scorer-only environment:

```
pip install numpy scipy pandas pyarrow matplotlib pillow pytest
```

1. Prove the scorers are healthy (they are unit-tested against synthetic
   trajectories with known outcomes):

   ```
   pytest
   ```

2. Generate the bundled example submission (closed-form analytic motions,
   not a simulator) and score it:

   ```
   python make_example.py
   python run_all.py --results-dir results/example
   ```

3. Read `results/example/report.md`. You should see 9 PASS and 9 NA. The
   closed-form example exercises the nine core trajectory/state scorers and
   intentionally omits thermal, bimaterial, native-body, triangle-state, and
   measured-material submissions, demonstrating explicit NA reporting.

## Submitting your own system

1. Read `SPEC.md`. It is short and normative: trajectory format (parquet or
   CSV), covariance format for Tier C, scene format, statuses.
2. For each scene JSON in `scenes/`, load the frozen initial positions from
   the scene's `.npy` file (row index = particle_id), set up the material and
   boundary conditions the scene describes, run for `duration_s`, and dump
   frames at `output_fps` into `results/<your_system>/<scene_id>/`.
3. Score everything:

   ```
   python run_all.py --results-dir results/<your_system>
   ```

4. Submit the results directory (or just `report.json` plus `meta.json`
   files) with your system name. Declare every deviation from a scene spec in
   that scene's `meta.json` notes; an undeclared deviation discovered later
   invalidates the entry.

Skip any scene your system cannot run (for example, the thermal scenes if you
have no heat transport). The report marks it NA, which is not a failure.

## The tests

| id | tier | class | what it checks | reference |
|---|---|---|---|---|
| M1 material validation | A | **model_validation** | uniaxial stress-strain vs a REAL measured natural rubber (ASTM D412 Yeoh); isolates the constitutive law, no boundary confound | measured tensile data |
| A1a in-place conservation | A | code_verification | momentum drift of a spinning body held in place; isolates the transfer scheme, tight tolerances | conservation laws |
| A1b drifting conservation | A | code_verification | momentum drift of the same body translating 11.5 m; transfer plus domain handling, looser tolerances, requires a domain_handling declaration in meta.json | conservation laws |
| A2 beam frequency | A | solution_verification | fundamental frequency of a plucked cantilever | Euler-Bernoulli |
| A3 wave speed | A | solution_verification | longitudinal first-arrival speed in a thin bar | c = sqrt(E/rho) |
| A4 melt front | A | capability (NA allowed) | melt front grows as sqrt(t) with the right prefactor | Neumann/Stefan solution |
| A5 stiffness | A | solution_verification | Poisson contraction under displacement control | eps_lat = -nu eps_axial |
| A6 free fall | A | solution_verification | center-of-mass position/velocity and deformation health on a native Gaussian MPM body | constant-gravity trajectory |
| B1 bimaterial bar | B | solution_verification | per-particle constitutive dispatch and interface coupling | series-bar strain jump + traction continuity |
| C1 covariance | C | code_verification | covariances transform as F Sigma F^T | affine push-forward |
| C2 covariance transport and health | C | code verification + capability | exact peak-shear covariance transport, collapse, and condition blowup | $F\Sigma_0F^\top$ plus boundedness |
| C3 round trip | C | capability | before/after renders match across an elastic squash | PSNR + SSIM |
| C4 captured geometry | C | capability | A2 on an irregular captured-style asset | measured-section Euler-Bernoulli |
| C5 triangle shape transport | C | capability | stable triangle edge and area transport under substantial motion | frame-zero triangle geometry |
| C6 rigid transport | C | code_verification | Galilean transport: uniform velocity in zero gravity stays straight, constant-speed, and shape-preserving | uniform motion + frame-zero shape |
| C7 supported settle | C | capability | long-horizon rest stability on the system's own released static support | static equilibrium after settling |
| C8 impact recovery | C | capability | soft-elastic floor impact with a deformation gate: bounded transient, finite state, tail recovery | elastic recovery of frame-zero shape |

Every scorer is a pure function of files, every tolerance lives in the scene
JSON (frozen per scene version), and every reference formula in `reference/`
carries its regime assumptions and a citation.

## Held-out variants

```
python variants/generate.py --scene a2_beam_frequency_v1 --seed 7
```

randomizes stiffness, density, and length within documented ranges and
regenerates the frozen positions; scorers recompute the reference from the
scaled scene at scoring time, so tuning to the public scene instance buys
nothing. Evaluation seeds are not committed.

## Repository layout

```
SPEC.md          the normative formats
scenes/          frozen scene JSONs + initial positions (.npy) + generator
scoring/         one scorer per test + common.py (loading, validation, reports)
reference/       pure-numpy analytical references with citations
variants/        held-out variant generator (output gitignored)
interventions/   counterfactual driver + frozen I1/I2 scene files
results/         submissions land here (gitignored except the example)
run_all.py       scores a results directory, writes report.md + report.json
make_example.py  regenerates the bundled example submission
tests/           unit tests for the scorers themselves
reports/         frozen lightweight reports for the paper entries
reference_entrant/gaussianflesh/
                 full-contract reference implementation and export harness
```

Note on the bundled example: the large `trajectory.parquet` files are
gitignored to keep the repository small; `python make_example.py` rebuilds
them deterministically in seconds.

## Changelog

Headline of **v0.6.0**: six external broad profiles are accompanied by a
uniform adapter-validation protocol, exact runner snapshots, compatibility
patches, matched negative controls, and tolerance-sensitivity reports. A6 and
the representation-native C5--C8 scenes extend coverage without inventing
unsupported state channels.

Headline of **v0.4.0**: Tier B adds the analytical B1 bimaterial-bar test for
per-particle constitutive dispatch and shared-grid interface coupling. The
central half of each material supplies the strain gauge, and all tolerances are
frozen in the scene.

Headline of **v0.3.0**: the `model_validation` class is no longer empty. `M1`
validates an entrant's material against a *real measured* natural rubber
(ASTM D412 tensile, Yeoh fit) in uniaxial tension  -  chosen because it isolates
the constitutive law with no clamp/bending confound. GaussianFlesh reproduces
the measured rubber to a 5.4% median error up to 150% strain. See
`CHANGELOG.md`.

Prior v0.2.0 headline: the original A1 conservation scene
conflated transfer-scheme conservation with domain handling of a
far-translating body, so it was split into A1a (in-place, tight, isolates
the transfer) and A1b (drifting, compound, requires a domain_handling
declaration). The old scene is preserved under `scenes/deprecated/`.

## License and citation

GaussianBench is released under the MIT License. See [`LICENSE`](LICENSE).
Citation metadata is provided in [`CITATION.cff`](CITATION.cff); update the
record with the final arXiv identifier when it becomes available.
