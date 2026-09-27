# GaussianBench report

**Tier A**: 0 pass, 0 fail, 8 n/a
**Tier B**: 0 pass, 0 fail, 1 n/a
**Tier C**: 0 pass, 1 fail, 8 n/a

| scene | class | tier | status | measured | reference | tolerance |
|---|---|---|---|---|---|---|
| a1a_conservation_inplace_v1 | code_verification | A | **NA** |  |  |  |
| a1b_conservation_drifting_v1 | code_verification | A | **NA** |  |  |  |
| a2_beam_frequency_v1 | solution_verification | A | **NA** |  |  |  |
| a3_wavespeed_v1 | solution_verification | A | **NA** |  |  |  |
| a4_meltfront_v1 | capability | A | **NA** |  |  |  |
| a5_stiffness_v1 | solution_verification | A | **NA** |  |  |  |
| a6_freefall_v1 | solution_verification | A | **NA** |  |  |  |
| b1_bimaterial_bar_v1 | solution_verification | B | **NA** |  |  |  |
| c1_covariance_v1 | code_verification | C | **FAIL** | kinematic_median_rel_frobenius_error=0.008656; kinematic_p95_rel_frobenius_error=0.01563; transport_median_rel_frobenius_error=0.5074; transport_p95_rel_frobenius_error=0.5303; kinematic_status=PASS; transport_status=FAIL; n_particles_scored=3375 | kinematic_target=F_fit from positions vs F_target; transport_map=F_native Sigma_0 F_native^T; F_target_diag=[1.5, 1.0, 0.8] | kinematic_median_max=0.05; kinematic_p95_max=0.1; transport_median_max=0.001; transport_p95_max=0.01 |
| c2_blobhealth_melt_v1 | capability | C | **NA** |  |  |  |
| c2_blobhealth_shear_v1 | capability | C | **NA** |  |  |  |
| c3_roundtrip_v1 | capability | C | **NA** |  |  |  |
| c4_captured_frequency_v1 | capability | C | **NA** |  |  |  |
| c5_triangle_shape_transport_v1 | capability | C | **NA** |  |  |  |
| c6_rigid_transport_v1 | code_verification | C | **NA** |  |  |  |
| c7_supported_settle_v1 | capability | C | **NA** |  |  |  |
| c8_impact_recovery_v1 | capability | C | **NA** |  |  |  |
| m1_material_validation_v1 | model_validation | A | **NA** |  |  |  |

### a1a_conservation_inplace_v1
- No submission directory found. For capability tests this means the system does not support the capability.

### a1b_conservation_drifting_v1
- No submission directory found. For capability tests this means the system does not support the capability.

### a2_beam_frequency_v1
- No submission directory found. For capability tests this means the system does not support the capability.

### a3_wavespeed_v1
- No submission directory found. For capability tests this means the system does not support the capability.

### a4_meltfront_v1
- No submission directory found. For capability tests this means the system does not support the capability.

### a5_stiffness_v1
- No submission directory found. For capability tests this means the system does not support the capability.

### a6_freefall_v1
- No submission directory found. For capability tests this means the system does not support the capability.

### b1_bimaterial_bar_v1
- No submission directory found. For capability tests this means the system does not support the capability.

### c1_covariance_v1
- Excluded 4625 particles within 2 spacings of a prescribed face; fitted and scored local F on 3375 particles; transport uses the entrant's exported internal F. Kinematics pass; pure covariance transport fails.

### c2_blobhealth_melt_v1
- No submission directory found. For capability tests this means the system does not support the capability.

### c2_blobhealth_shear_v1
- No submission directory found. For capability tests this means the system does not support the capability.

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
