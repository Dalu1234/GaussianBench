# GaussianBench report

**Tier A**: 5 pass, 1 fail, 2 n/a
**Tier B**: 0 pass, 0 fail, 1 n/a
**Tier C**: 2 pass, 1 fail, 5 n/a, 1 invalid/error

| scene | class | tier | status | measured | reference | tolerance |
|---|---|---|---|---|---|---|
| a1a_conservation_inplace_v1 | code_verification | A | **FAIL** | linear_momentum_rel_drift=8.948e-06; angular_momentum_rel_drift=7.298e-05; kinetic_energy_rel_change_final=-1.696e-04 | linear_momentum_rel_drift=0; angular_momentum_rel_drift=0 | linear_momentum_rel_drift=1.000e-06; angular_momentum_rel_drift=0.001; velocities_derived=False |
| a1b_conservation_drifting_v1 | code_verification | A | **PASS** | linear_momentum_rel_drift=1.023e-06; angular_momentum_rel_drift=6.954e-05; kinetic_energy_rel_change_final=-9.645e-07; declared_domain_handling=co_moving | linear_momentum_rel_drift=0; angular_momentum_rel_drift=0 | linear_momentum_rel_drift=0.001; angular_momentum_rel_drift=0.01; velocities_derived=False |
| a2_beam_frequency_v1 | solution_verification | A | **PASS** | fundamental_frequency_hz=1.177; rel_error=0.04228; damping_ratio=0.005628; max_tip_deflection_m=0.007002 | fundamental_frequency_hz=1.129 | rel_tolerance=0.1 |
| a3_wavespeed_v1 | solution_verification | A | **PASS** | bar_wave_speed_m_s=24.09; arrival_time_s=0.041; travel_distance_m=0.9875; rel_error=0.07713 | bar_wave_speed_m_s=22.36 | rel_tolerance=0.15 |
| a4_meltfront_v1 | capability | A | **NA** |  |  |  |
| a5_stiffness_v1 | solution_verification | A | **PASS** | lateral_strain=-0.00645; axial_strain=0.02104; rel_error=0.02174 | lateral_strain=-0.006313; poisson_ratio=0.3 | rel_tolerance=0.1 |
| a6_freefall_v1 | solution_verification | A | **NA** |  |  |  |
| b1_bimaterial_bar_v1 | solution_verification | B | **NA** |  |  |  |
| c1_covariance_v1 | code_verification | C | **PASS** | kinematic_median_rel_frobenius_error=0.008656; kinematic_p95_rel_frobenius_error=0.01563; transport_median_rel_frobenius_error=1.363e-04; transport_p95_rel_frobenius_error=3.375e-04; kinematic_status=PASS; transport_status=PASS; n_particles_scored=3375 | kinematic_target=F_fit from positions vs F_target; transport_map=F_native Sigma_0 F_native^T; F_target_diag=[1.5, 1.0, 0.8] | kinematic_median_max=0.05; kinematic_p95_max=0.1; transport_median_max=0.001; transport_p95_max=0.01 |
| c2_blobhealth_melt_v1 | capability | C | **NA** |  |  |  |
| c2_blobhealth_shear_v1 | capability | C | **INVALID** | peak_kinematic_median_rel_frobenius_error=0.453; peak_kinematic_p95_rel_frobenius_error=0.6461; peak_transport_median_rel_frobenius_error=7.469e-04; peak_transport_p95_rel_frobenius_error=0.002806; kinematic_status=FAIL; transport_status=PASS; health_status=PASS; min_eigenvalue_ratio=0.1416; degenerate_fraction=0; worst_particle_id=996 | kinematic_target=F_fit from positions vs F_peak; peak_covariance_map=F_native Sigma_0 F_native^T; peak_gamma=1; healthy_min_eigenvalue_ratio=1; healthy_degenerate_fraction=0 | peak_kinematic_median_rel_frobenius_max=0.05; peak_kinematic_p95_rel_frobenius_max=0.1; peak_transport_median_rel_frobenius_max=0.001; peak_transport_p95_rel_frobenius_max=0.01; peak_transport_boundary_exclusion_spacings=2; min_eigenvalue_ratio=0.001; condition_number_max=1.000e+04; degenerate_fraction_max=0.001 |
| c3_roundtrip_v1 | capability | C | **FAIL** | psnr_db=31.63; ssim=0.9922; min_height_frac_during_run=0.8 | psnr_db=inf (identical renders); ssim=1 | psnr_min_db=35; ssim_min=0.97 |
| c4_captured_frequency_v1 | capability | C | **PASS** | fundamental_frequency_hz=0.9137; rel_error=0.006905; damping_ratio=0.007951 | fundamental_frequency_hz=0.9074; section_I_over_A_m2=1.317e-04 | rel_tolerance=0.2 |
| c5_triangle_shape_transport_v1 | capability | C | **NA** |  |  |  |
| c6_rigid_transport_v1 | code_verification | C | **NA** |  |  |  |
| c7_supported_settle_v1 | capability | C | **NA** |  |  |  |
| c8_impact_recovery_v1 | capability | C | **NA** |  |  |  |
| m1_material_validation_v1 | model_validation | A | **PASS** | median_rel_error=0.05397; p95_rel_error=0.08729; max_stretch_scored=2.5; max_strain_scored=1.5; n_points=45 | source=Azarniya & Rahimi 2022, ASTM D412 Yeoh fit | median_rel_tolerance=0.15; max_stretch=2.5 |

### a1a_conservation_inplace_v1
- Angular momentum magnitude trace |L(t)|/|L(0)|: {'t=0.00s': 1.0, 't=2.50s': 1.0000284372283261, 't=5.00s': 1.0000352086726594, 't=7.50s': 1.0000458742259324, 't=10.00s': 1.0000521101365683}

### a1b_conservation_drifting_v1
- Angular momentum magnitude trace |L(t)|/|L(0)|: {'t=0.00s': 1.0, 't=2.50s': 1.0000317650756476, 't=5.00s': 1.0000371046412204, 't=7.50s': 1.0000387293759816, 't=10.00s': 1.0000426509524694}

### a2_beam_frequency_v1
- Damping ratio (report-only, numerical dissipation): zeta = 0.005628.
- Young's modulus used directly in the Euler-Bernoulli formula (uniaxial stress, lateral faces free; no E/(1-nu^2) correction).

### a3_wavespeed_v1
- First motion of the far-end set (mean of 64 particles) at frame 82, t = 0.0410 s.

### a4_meltfront_v1
- No submission directory found. For capability tests this means the system does not support the capability.

### a5_stiffness_v1
- Axial strain at hold: 0.02104 (from measured face displacements over the rest gauge length 0.0950 m).
- Lateral strain measured on core region x in [30%, 70%] of length: eps_y=-0.00645, eps_z=-0.00645.
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
- Peak frame 60: median/p95 kinematic realization error 4.530e-01/6.461e-01; pure covariance transport error 7.469e-04/2.806e-03 using entrant-native F on 180 interior particles after excluding 2 spacings from each face. Worst health particle id 996; the scorer regenerates the eigenvalue history when the full native export is rescored.

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
- Uniaxial tension isolates the constitutive law (no clamp, no bending), so this gap is cleanly attributable to the material model.
- Compared against the measured natural-rubber curve up to 2.50x stretch (150% engineering strain).
