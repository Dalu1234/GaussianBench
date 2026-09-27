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
| c1_covariance_v1 | code_verification | C | **NA** |  |  |  |
| c2_blobhealth_melt_v1 | capability | C | **NA** |  |  |  |
| c2_blobhealth_shear_v1 | capability | C | **NA** |  |  |  |
| c3_roundtrip_v1 | capability | C | **NA** |  |  |  |
| c4_captured_frequency_v1 | capability | C | **NA** |  |  |  |
| c5_triangle_shape_transport_v1 | capability | C | **NA** |  |  |  |
| c6_rigid_transport_v1 | code_verification | C | **NA** |  |  |  |
| c7_supported_settle_v1 | capability | C | **NA** |  |  |  |
| c8_impact_recovery_v1 | capability | C | **FAIL** | triangle_count=448606; max_com_travel=6.849; peak_area_rel_error_p95=0.8596; max_edge_rel_error_p95=0.287; tail_best_edge_rel_error_median=0.02253; tail_best_area_rel_error_median=0.1272; max_nonfinite_fraction=0 | tail_shape_error=returns below the recovery bound; transient=bounded, finite, identity-stable | transient_edge_p95_max=1; tail_edge_median_max=0.05; tail_area_median_max=0.1 |
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
- No submission directory found. For capability tests this means the system does not support the capability.

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

### m1_material_validation_v1
- No submission directory found. For capability tests this means the system does not support the capability.
