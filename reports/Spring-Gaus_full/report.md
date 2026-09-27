# GaussianBench report

**Tier A**: 0 pass, 2 fail, 6 n/a
**Tier B**: 0 pass, 0 fail, 1 n/a
**Tier C**: 0 pass, 0 fail, 9 n/a

| scene | class | tier | status | measured | reference | tolerance |
|---|---|---|---|---|---|---|
| a1a_conservation_inplace_v1 | code_verification | A | **FAIL** | linear_momentum_rel_drift=5.372e+07; angular_momentum_rel_drift=0.3736; kinetic_energy_rel_change_final=42.37 | linear_momentum_rel_drift=0; angular_momentum_rel_drift=0 | linear_momentum_rel_drift=1.000e-06; angular_momentum_rel_drift=0.001; velocities_derived=False |
| a1b_conservation_drifting_v1 | code_verification | A | **FAIL** | linear_momentum_rel_drift=0.05318; angular_momentum_rel_drift=0.1973; kinetic_energy_rel_change_final=0.4316; declared_domain_handling=mesh_free | linear_momentum_rel_drift=0; angular_momentum_rel_drift=0 | linear_momentum_rel_drift=0.001; angular_momentum_rel_drift=0.01; velocities_derived=False |
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
| c8_impact_recovery_v1 | capability | C | **NA** |  |  |  |
| m1_material_validation_v1 | model_validation | A | **NA** |  |  |  |

### a1a_conservation_inplace_v1
- Angular momentum magnitude trace |L(t)|/|L(0)|: {'t=0.00s': 1.0, 't=2.50s': 0.9798053392977297, 't=5.00s': 0.9510473743096588, 't=7.50s': 1.2020365698445385, 't=10.00s': 1.3069499790148837}

### a1b_conservation_drifting_v1
- Angular momentum magnitude trace |L(t)|/|L(0)|: {'t=0.00s': 1.0, 't=2.50s': 0.9798126459806186, 't=5.00s': 0.9511028463155251, 't=7.50s': 1.1277597408000386, 't=10.00s': 1.0935647298625908}

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

### c8_impact_recovery_v1
- No submission directory found. For capability tests this means the system does not support the capability.

### m1_material_validation_v1
- No submission directory found. For capability tests this means the system does not support the capability.
