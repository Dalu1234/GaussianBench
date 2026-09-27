# Adapter-validation protocol

GaussianBench treats an adapter as part of the experimental apparatus.  Every
reported external entry is audited with the same six checks before its verdicts
are used in a cross-system claim.

1. **Release pin and provenance.** Record the upstream commit, benchmark/harness
   commit, hardware, and exact native entry point in `meta.json`.
2. **State and identity round trip.** Preserve stable native identities, verify
   finite rows and frame-zero geometry in benchmark coordinates, and record every
   axis, scale, offset, and moving-frame transform.  Exact inverse transforms are
   applied on export.
3. **Parameter and unit parity.** Map scene parameters through the release's own
   parameter functions.  For continuum solvers, compare the resulting density and
   Lame parameters with the scene values.  If no released mapping exists, the
   scene is NA rather than supplied by the adapter.
4. **Scene realization.** Measure prescribed faces, initial velocity, gravity,
   contact, or deformation gates from the exported state.  A missed prescription
   is INVALID, not a solver FAIL.
5. **Native-state trace.** Export positions, deformation, covariance, or native
   elements from the same fields used by the released solver/renderer.  The
   adapter may format state but may not reconstruct an unsupported state channel.
6. **Matched negative control.** Apply one localized, known fault without changing
   unrelated state.  The intended metric must change verdict or cross its stated
   threshold while orthogonal metrics remain unchanged or healthy.

The audit is capability-aware.  Native triangle systems are not required to
invent particle deformation gradients; spring systems are not assigned continuum
material scenes without a released mapping.  `NA` therefore passes the honesty
part of the protocol, but it does not increase benchmark coverage.

## Applied audit

| System | Profile | Pin/provenance | State/ID round trip | Parameter/unit parity | Realization audit | Native-state trace | Matched negative control |
|---|---|---|---|---|---|---|---|
| PhysGaussian | Broad | Verified | Verified | Verified | Verified | Verified (`F`, covariance) | Frozen covariance C1 |
| GaussianFluent | Broad | Verified | Verified | Verified | Verified | Verified (`F`, covariance) | Frozen covariance C1 |
| OmniPhysGS | Broad | Verified | Verified | Verified | Verified | Verified (`F`, covariance) | Frozen covariance C1 |
| PhysDreamer | Broad | Verified | Verified | Verified | Verified | Verified (`F`, covariance) | Frozen covariance C1 |
| Physics3D | Broad | Verified | Verified | Verified (`E/10^7`) | Verified | Verified (`F`, renderer covariance) | Half gravity A6; frozen covariance C1 |
| GASP | Broad | Verified | Verified (stable triangles) | Native released controls | Verified | Verified (triangle state; native `F` in A6) | Half gravity A6; gravity C6; sand C8 |
| Spring-Gaus | Targeted | Verified | Verified (spring points) | NA outside conservation | Verified where entered | Positions/velocities only | Conservation failure is observed, not used as a matched fault |
| PhysTwin | Targeted | Verified | Verified (spring points) | NA outside native A6 | Verified where entered | Positions/velocities only | Half-gravity intervention unavailable because gravity is hard-coded |

The broad-entry fault directories are machine-readable under `results/`.  The two
targeted probes are retained to document interface breadth but are not used for
the paper's cross-system benchmark claim.

## Reviewer evidence trail

The audit labels above are backed by inspectable files rather than self-attested
checkboxes. For each system, reviewers should cross-check:

- the exact runner and compatibility patch under `adapters/reproduction/`;
- the frozen scene JSON under `scenes/`;
- per-scene `meta.json` under the supplementary package's `reports/` directory;
- the measured verdict in that profile's `report.json`;
- the matched fault profile named in the table above.

Compatibility patches are classified explicitly. Import fallbacks, optional
visualization, and public-API substitutions are execution-only changes.
Physics3D additionally exposes a selectable base-corotated branch and repairs
identity initialization of a persistent released field; its complete patch is
included because that change deserves direct scrutiny. No patch changes a
GaussianBench reference, tolerance, scorer, or frozen scene.
