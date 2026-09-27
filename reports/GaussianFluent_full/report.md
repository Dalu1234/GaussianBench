# GaussianBench report

**Tier A**: 1 pass, 4 fail, 3 n/a
**Tier B**: 0 pass, 0 fail, 1 n/a
**Tier C**: 1 pass, 0 fail, 7 n/a, 1 invalid/error

| scene | class | tier | status | measured | reference | tolerance |
|---|---|---|---|---|---|---|
| a1a_conservation_inplace_v1 | code_verification | A | **FAIL** | linear_momentum_rel_drift=0.01725; angular_momentum_rel_drift=1.019; kinetic_energy_rel_change_final=-0.9997 | linear_momentum_rel_drift=0; angular_momentum_rel_drift=0 | linear_momentum_rel_drift=1.000e-06; angular_momentum_rel_drift=0.001; velocities_derived=False |
| a1b_conservation_drifting_v1 | code_verification | A | **FAIL** | linear_momentum_rel_drift=0.003037; angular_momentum_rel_drift=1.019; kinetic_energy_rel_change_final=-0.008343; declared_domain_handling=co_moving | linear_momentum_rel_drift=0; angular_momentum_rel_drift=0 | linear_momentum_rel_drift=0.001; angular_momentum_rel_drift=0.01; velocities_derived=False |
| a2_beam_frequency_v1 | solution_verification | A | **FAIL** | fundamental_frequency_hz=0.2607; rel_error=0.7691; damping_ratio=0.2111; max_tip_deflection_m=0.01534 | fundamental_frequency_hz=1.129 | rel_tolerance=0.1 |
| a3_wavespeed_v1 | solution_verification | A | **FAIL** | arrival_detected=False | bar_wave_speed_m_s=22.36 | rel_tolerance=0.15 |
| a4_meltfront_v1 | capability | A | **NA** |  |  |  |
| a5_stiffness_v1 | solution_verification | A | **PASS** | lateral_strain=-0.006297; axial_strain=0.02104; rel_error=0.002628 | lateral_strain=-0.006313; poisson_ratio=0.3 | rel_tolerance=0.1 |
| a6_freefall_v1 | solution_verification | A | **NA** |  |  |  |
| b1_bimaterial_bar_v1 | solution_verification | B | **NA** |  |  |  |
| c1_covariance_v1 | code_verification | C | **PASS** | kinematic_median_rel_frobenius_error=0.008643; kinematic_p95_rel_frobenius_error=0.01175; transport_median_rel_frobenius_error=1.119e-04; transport_p95_rel_frobenius_error=3.136e-04; kinematic_status=PASS; transport_status=PASS; n_particles_scored=3375 | kinematic_target=F_fit from positions vs F_target; transport_map=F_native Sigma_0 F_native^T; F_target_diag=[1.5, 1.0, 0.8] | kinematic_median_max=0.05; kinematic_p95_max=0.1; transport_median_max=0.001; transport_p95_max=0.01 |
| c2_blobhealth_melt_v1 | capability | C | **NA** |  |  |  |
| c2_blobhealth_shear_v1 | capability | C | **INVALID** | peak_kinematic_median_rel_frobenius_error=0.1237; peak_kinematic_p95_rel_frobenius_error=0.1997; peak_transport_median_rel_frobenius_error=8.146e-04; peak_transport_p95_rel_frobenius_error=0.001465; kinematic_status=FAIL; transport_status=PASS; health_status=PASS; min_eigenvalue_ratio=0.18; degenerate_fraction=0; worst_particle_id=999 | kinematic_target=F_fit from positions vs F_peak; peak_covariance_map=F_native Sigma_0 F_native^T; peak_gamma=1; healthy_min_eigenvalue_ratio=1; healthy_degenerate_fraction=0 | peak_kinematic_median_rel_frobenius_max=0.05; peak_kinematic_p95_rel_frobenius_max=0.1; peak_transport_median_rel_frobenius_max=0.001; peak_transport_p95_rel_frobenius_max=0.01; peak_transport_boundary_exclusion_spacings=2; min_eigenvalue_ratio=0.001; condition_number_max=1.000e+04; degenerate_fraction_max=0.001 |
| c3_roundtrip_v1 | capability | C | **NA** |  |  |  |
| c4_captured_frequency_v1 | capability | C | **NA** |  |  |  |
| c5_triangle_shape_transport_v1 | capability | C | **NA** |  |  |  |
| c6_rigid_transport_v1 | code_verification | C | **NA** |  |  |  |
| c7_supported_settle_v1 | capability | C | **NA** |  |  |  |
| c8_impact_recovery_v1 | capability | C | **NA** |  |  |  |
| m1_material_validation_v1 | model_validation | A | **NA** |  |  |  |

### a1a_conservation_inplace_v1
- Angular momentum magnitude trace |L(t)|/|L(0)|: {'t=0.00s': 1.0, 't=2.50s': 0.0030264414781387863, 't=5.00s': 0.0022821757538660685, 't=7.50s': 0.001684384108081549, 't=10.00s': 0.0011213366111166865}

### a1b_conservation_drifting_v1
- Angular momentum magnitude trace |L(t)|/|L(0)|: {'t=0.00s': 1.0, 't=2.50s': 0.0030269333073355974, 't=5.00s': 0.002279198378290836, 't=7.50s': 0.0017040200403225549, 't=10.00s': 0.001046064733513582}

### a2_beam_frequency_v1
- Damping ratio (report-only, numerical dissipation): zeta = 0.2111.
- Young's modulus used directly in the Euler-Bernoulli formula (uniaxial stress, lateral faces free; no E/(1-nu^2) correction).

### a3_wavespeed_v1
- The far end never moved above the noise floor (1e-05 m) within the run, or moved at frame 0 (which would mean the initial condition was not at rest). No wave arrived; the pulse boundary condition may be missing.

### a4_meltfront_v1
- No submission directory found. For capability tests this means the system does not support the capability.

### a5_stiffness_v1
- Axial strain at hold: 0.02104 (from measured face displacements over the rest gauge length 0.0950 m).
- Lateral strain measured on core region x in [30%, 70%] of length: eps_y=-0.00630, eps_z=-0.00630.
- Reminder: lateral faces must be traction-free (uniaxial stress). Near-zero contraction usually means your lateral boundaries are clamped; declare boundary handling in meta.json.

### a6_freefall_v1
- No submission directory found. For capability tests this means the system does not support the capability.

### b1_bimaterial_bar_v1
- No submission directory found. For capability tests this means the system does not support the capability.

### c1_covariance_v1
- Excluded 4625 particles within 2 spacings of a prescribed face; fitted and scored local F on 3375 particles; transport uses the entrant's exported internal F. Kinematics pass; pure covariance transport passes.

### c2_blobhealth_melt_v1
- No submission directory found. For capability tests this means the system does not support the capability.

### c2_blobhealth_shear_v1
- Peak frame 60: median/p95 kinematic realization error 1.237e-01/1.997e-01; pure covariance transport error 8.146e-04/1.465e-03 using entrant-native F on 180 interior particles after excluding 2 spacings from each face. Worst health particle id 999; the scorer regenerates the eigenvalue history when the full native export is rescored.

### c3_roundtrip_v1
- No submission directory found. For capability tests this means the system does not support the capability.

### c4_captured_frequency_v1
- No submission directory found. For capability tests this means the system does not support the capability.

### c5_triangle_shape_transport_v1
- No submission directory found. For capability tests this means the system does not support the capability.

### c6_rigid_transport_v1
- No submission directory found. For capability tests this means the system does not support the capability.

### c7_supported_settle_v1
- No submission directory found. For capability tests this means the system does not support the capability.

### c8_impact_recovery_v1
- No submission directory found. For capability tests this means the system does not support the capability.

### m1_material_validation_v1
- No submission directory found. For capability tests this means the system does not support the capability.
