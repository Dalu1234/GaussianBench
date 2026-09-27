# Changelog

## v0.6.0

### Added
- Three representation-native shape scenes sharing the frozen
  `triangle_metrics.csv` contract (SPEC 2.6), designed so stable-identity
  systems beyond triangle soups can also enter by declaring their element
  mapping:
  - `c6_rigid_transport_v1` (code_verification): Galilean rigid transport.
    Uniform velocity in zero gravity along a grid-oblique direction; the
    center of mass must advect straight at constant speed while shape metrics
    stay at rest values. Natural fault: engine gravity left enabled.
  - `c7_supported_settle_v1` (capability): long-horizon rest stability on the
    system's own released static support under native gravity, with a
    body-scaled post-settling drift bound (requires `bbox_volume`). Natural
    fault: over-aggressive shape-control threshold under sustained contact.
  - `c8_impact_recovery_v1` (capability): soft-elastic floor impact with a
    deformation gate (too-gentle runs are INVALID), bounded transient, and
    tail recovery bounds. Natural fault: released inelastic material (no
    recovery).
- Synthetic PASS/FAIL/INVALID scorer regression tests for all three scenes,
  including each scene's natural controlled fault.

## v0.5.0

### Added
- A6 short-horizon free-fall verification for native Gaussian MPM bodies,
  grading center-of-mass position and velocity together with bounding-volume
  and deformation-gradient health.
- C5 representation-native triangle shape transport, with a motion gate and
  stable-identity edge and area metrics.
- Released adapters and frozen reports for Physics3D and GASP. The native
  entries pass their applicable scenes; matched half-gravity and shape-
  threshold interventions fail the same scorers.

## v0.4.0

### Added
- **Tier B relational verification** and its first scene,
  `b1_bimaterial_bar_v1`.
- B1 assigns a soft corotated law and a stiff stable-Neo-Hookean law to two
  halves of one connected bar.  Under a small prescribed extension, the
  scorer checks the analytical series-bar strain jump in both regions and
  continuity of the inferred interface traction.
- Regional strain is measured over the central half of each material,
  symmetrically excluding the clamp- and interface-adjacent quarters.
- The scene closes the previous evaluation gap in which per-particle
  constitutive dispatch and shared-grid mixed-material coupling were shown
  only qualitatively.
- Synthetic PASS/FAIL/INVALID scorer regression tests and a three-resolution
  GaussianFlesh convergence driver.
- Two paired intervention protocols: I1 changes only the A4 hot-floor
  temperature and grades the melt-prefactor ratio; I2 changes only B1
  per-particle material painting and grades the analytical extension share.

## v0.3.0

### Added
- **First `model_validation` entry** (`m1_material_validation_v1`): the suite's
  first test against MEASURED physical data rather than an analytical
  reference. A natural-rubber specimen characterized independently by an
  ASTM D412 uniaxial tensile test (Azarniya & Rahimi 2022, Yeoh fit) is the
  reference; the entrant submits its own uniaxial Cauchy stress vs stretch
  (`stress_strain.csv`), and the scorer compares to the measured curve. This
  fills the previously-empty `model_validation` class.
- Uniaxial tension is the deliberate choice: no clamp, no bending, no
  boundary-condition confound, so the gap is cleanly attributable to the
  constitutive law. (A companion investigation confirmed why a structural
  target is unsuitable: on this same specimen a clamped-clamped beam is ~45%
  stiffer than ideal beam theory purely from clamp-fixture effects the data
  does not pin down, so a sim-vs-experiment gap there is un-attributable.)
- `reference/yeoh_rubber.py` (measured-rubber Yeoh reference with citation),
  `scoring/score_m1_material.py`, and scorer unit tests.
- First recorded entrant result: GaussianFlesh reproduces the measured
  rubber's uniaxial stress-strain to a 5.4% median error up to 150% strain
  (`results/GaussianFlesh_material_2026-07-08/`).

### Note
- The material scene uses a `stress_strain.csv` submission (columns `stretch`,
  `cauchy_stress_pa`) rather than a trajectory, since a constitutive
  comparison needs stress, which the trajectory format does not carry.

## v0.2.1

### Fixed
- Conservation scorer erratum: angular momentum is now measured as SPIN
  angular momentum, about the instantaneous center of mass and relative
  to the center-of-mass velocity. The v0.2.0 definition (about the
  initial center of mass) was contaminated by allowed linear-momentum
  drift through the cross term M (x_cm(t) - x_cm0) x v_cm: on the
  drifting A1b scene, a p drift merely AT its own 1e-3 tolerance induced
  an apparent L drift of order 3 (300x the L tolerance), so the L metric
  was double-counting the p drift instead of measuring rotation. Spin
  angular momentum is equally conserved for an isolated body and immune
  to the cross term. Discovered by the first entrant's A1b submission,
  whose sim-frame spin was conserved to 5e-4 while the reported L drift
  read 0.56. Tolerances are unchanged; scene provenance formulas carry
  the erratum note, and a regression test pins the fixed behavior.

## v0.2.0

### Changed
- Split the A1 conservation scene into A1a (`a1a_conservation_inplace_v1`:
  in-place spin, tight tolerances, isolates the particle-grid transfer) and
  A1b (`a1b_conservation_drifting_v1`: the original drifting scene, looser
  tolerances, tests transfer plus domain handling as a declared compound
  quantity). The original scene is deprecated to `scenes/deprecated/`.
  Rationale: A1 v1 conflated two distinct diagnostic concerns and could
  fail a correct transfer scheme for domain-handling reasons alone (a
  fixed-grid entrant covering the 11.5 m drift envelope is forced to a
  grid so coarse the 0.05 m body becomes sub-grid). The split isolates the
  conflated diagnostics; it does not ease any test, because the original
  tight tolerances live on unchanged in A1a.
- A1a additionally promotes monotone kinetic energy growth beyond 1
  percent from a report-only flag to a FAIL: with the body held in place,
  sustained growth can only come from the integration scheme.
- Spec version 1.0.0 to 1.1.0: `meta.json` `notes` may now be a JSON
  object (free text under `notes.text`) to carry structured declarations.

### Added
- Generic `requires_meta_declaration` mechanism in scene `pass_criteria`:
  scenes can demand submitter declarations at dotted meta.json paths, with
  an allowed-value set; missing or invalid declarations score INVALID, and
  satisfied ones are echoed into the verdict for report grouping. First
  user: A1b's `notes.domain_handling` category
  (fixed / co_moving / sparse / mesh_free / other).
- `scenes/deprecated/` directory for superseded scenes, preserved with
  their frozen criteria and positions files for historical
  reproducibility, excluded from scene discovery.
- `CHANGELOG.md` and `VERSION` (this release: 0.2.0).

### Preserved
- All other v0.1.0 scenes and their frozen pass_criteria and provenance
  blocks are unchanged. Historical results scored against
  `a1_conservation_v1` remain valid records; that scene_id simply no
  longer matches a live scene.

## v0.1.0

Initial release: Tier A (A1 conservation, A2 beam frequency, A3 wave
speed, A4 melt front, A5 Poisson stiffness), Tier C (C1 covariance
push-forward, C2 blob health, C3 appearance round trip, C4 captured
geometry), pure-file scorers with scene-frozen tolerances, synthetic
ground-truth unit tests, held-out variant generator, bundled analytic
example submission.
