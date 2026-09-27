# GaussianBench report

**Tier A**: 3 pass, 1 fail, 4 n/a
**Tier B**: 0 pass, 1 fail, 0 n/a
**Tier C**: 1 pass, 1 fail, 6 n/a, 1 invalid/error

| scene | class | tier | status | measured | reference | tolerance |
|---|---|---|---|---|---|---|
| a1a_conservation_inplace_v1 | code_verification | A | **FAIL** | linear_momentum_rel_drift=4.164e-06; angular_momentum_rel_drift=3.511e-04; kinetic_energy_rel_change_final=-8.191e-04 | linear_momentum_rel_drift=0; angular_momentum_rel_drift=0 | linear_momentum_rel_drift=1.000e-06; angular_momentum_rel_drift=0.001; velocities_derived=False |
| a1b_conservation_drifting_v1 | code_verification | A | **NA** |  |  |  |
| a2_beam_frequency_v1 | solution_verification | A | **PASS** | fundamental_frequency_hz=1.22; rel_error=0.08056; damping_ratio=0.002436; max_tip_deflection_m=0.006812 | fundamental_frequency_hz=1.129 | rel_tolerance=0.1 |
| a3_wavespeed_v1 | solution_verification | A | **PASS** | bar_wave_speed_m_s=24.38; arrival_time_s=0.0405; travel_distance_m=0.9875; rel_error=0.09043 | bar_wave_speed_m_s=22.36 | rel_tolerance=0.15 |
| a4_meltfront_v1 | capability | A | **NA** |  |  |  |
| a5_stiffness_v1 | solution_verification | A | **PASS** | lateral_strain=-0.006832; axial_strain=0.02105; rel_error=0.08178 | lateral_strain=-0.006316; poisson_ratio=0.3 | rel_tolerance=0.1 |
| a6_freefall_v1 | solution_verification | A | **NA** |  |  |  |
| b1_bimaterial_bar_v1 | solution_verification | B | **FAIL** | left_strain=0.01955; right_strain=0.008631; left_strain_rel_error=0.1608; right_strain_rel_error=1.05; left_inferred_traction_pa=977.5; right_inferred_traction_pa=1726; interface_traction_jump_rel=0.5538; face_displacement_m=0.001 | left_strain=0.01684; right_strain=0.004211; continuous_traction_pa=842.1 | regional_strain_rel=0.15; interface_traction_jump_rel=0.15 |
| c1_covariance_v1 | code_verification | C | **FAIL** | kinematic_median_rel_frobenius_error=0.271; kinematic_p95_rel_frobenius_error=0.7361; transport_median_rel_frobenius_error=3.662e-08; transport_p95_rel_frobenius_error=6.361e-08; kinematic_status=FAIL; transport_status=PASS; n_particles_scored=3375 | kinematic_target=F_fit from positions vs F_target; transport_map=F_native Sigma_0 F_native^T; F_target_diag=[1.5, 1.0, 0.8] | kinematic_median_max=0.05; kinematic_p95_max=0.1; transport_median_max=0.001; transport_p95_max=0.01 |
| c2_blobhealth_melt_v1 | capability | C | **NA** |  |  |  |
| c2_blobhealth_shear_v1 | capability | C | **INVALID** | peak_kinematic_median_rel_frobenius_error=0.3689; peak_kinematic_p95_rel_frobenius_error=0.4027; peak_transport_median_rel_frobenius_error=4.415e-08; peak_transport_p95_rel_frobenius_error=6.394e-08; kinematic_status=FAIL; transport_status=PASS; health_status=PASS; min_eigenvalue_ratio=0.8432; degenerate_fraction=0; worst_particle_id=973 | kinematic_target=F_fit from positions vs F_peak; peak_covariance_map=F_native Sigma_0 F_native^T; peak_gamma=1; healthy_min_eigenvalue_ratio=1; healthy_degenerate_fraction=0 | peak_kinematic_median_rel_frobenius_max=0.05; peak_kinematic_p95_rel_frobenius_max=0.1; peak_transport_median_rel_frobenius_max=0.001; peak_transport_p95_rel_frobenius_max=0.01; peak_transport_boundary_exclusion_spacings=2; min_eigenvalue_ratio=0.001; condition_number_max=1.000e+04; degenerate_fraction_max=0.001 |
| c3_roundtrip_v1 | capability | C | **NA** |  |  |  |
| c4_captured_frequency_v1 | capability | C | **PASS** | fundamental_frequency_hz=0.9693; rel_error=0.06823; damping_ratio=0.002156 | fundamental_frequency_hz=0.9074; section_I_over_A_m2=1.317e-04 | rel_tolerance=0.2 |
| c5_triangle_shape_transport_v1 | capability | C | **NA** |  |  |  |
| c6_rigid_transport_v1 | code_verification | C | **NA** |  |  |  |
| c7_supported_settle_v1 | capability | C | **NA** |  |  |  |
| c8_impact_recovery_v1 | capability | C | **NA** |  |  |  |
| m1_material_validation_v1 | model_validation | A | **NA** |  |  |  |

### a1a_conservation_inplace_v1
- Angular momentum magnitude trace |L(t)|/|L(0)|: {'t=0.00s': 1.0, 't=2.50s': 0.9999507480223173, 't=5.00s': 0.9998616499500836, 't=7.50s': 0.9998660423560455, 't=10.00s': 0.9996815561032324}

### a1b_conservation_drifting_v1
- No submission directory found. For capability tests this means the system does not support the capability.

### a2_beam_frequency_v1
- Damping ratio (report-only, numerical dissipation): zeta = 0.002436.
- Young's modulus used directly in the Euler-Bernoulli formula (uniaxial stress, lateral faces free; no E/(1-nu^2) correction).

### a3_wavespeed_v1
- First motion of the far-end set (mean of 64 particles) at frame 81, t = 0.0405 s.

### a4_meltfront_v1
- No submission directory found. For capability tests this means the system does not support the capability.

### a5_stiffness_v1
- Axial strain at hold: 0.02105 (from measured face displacements over the rest gauge length 0.0950 m).
- Lateral strain measured on core region x in [30%, 70%] of length: eps_y=-0.00683, eps_z=-0.00683.
- Reminder: lateral faces must be traction-free (uniaxial stress). Near-zero contraction usually means your lateral boundaries are clamped; declare boundary handling in meta.json.

### a6_freefall_v1
- No submission directory found. For capability tests this means the system does not support the capability.

### b1_bimaterial_bar_v1
- Core-window strains use x/L in [0.125, 0.375] and [0.625, 0.875].
- The left and right halves use different constitutive IDs; passing requires both the analytical strain jump and shared-grid traction continuity.

### c1_covariance_v1
- Excluded 4625 particles within 2 spacings of a prescribed face; fitted and scored local F on 3375 particles; transport uses the entrant's exported internal F. Kinematics fail; pure covariance transport passes.

### c2_blobhealth_melt_v1
- No submission directory found. For capability tests this means the system does not support the capability.

### c2_blobhealth_shear_v1
- Peak frame 60: median/p95 kinematic realization error 3.689e-01/4.027e-01; pure covariance transport error 4.415e-08/6.394e-08 using entrant-native F on 180 interior particles after excluding 2 spacings from each face. Worst health particle id 973; the scorer regenerates the eigenvalue history when the full native export is rescored.

### c3_roundtrip_v1
- No submission directory found. For capability tests this means the system does not support the capability.

### c4_captured_frequency_v1
- Reference computed from measured section properties of the irregular asset (see scene provenance); tolerance is wider than A2 accordingly.

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
