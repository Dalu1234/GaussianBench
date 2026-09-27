# GaussianBench report

**Tier A**: 1 pass, 1 fail, 4 n/a
**Tier C**: 0 pass, 0 fail, 5 n/a

| scene | class | tier | status | measured | reference | tolerance |
|---|---|---|---|---|---|---|
| a1a_conservation_inplace_v1 | code_verification | A | **FAIL** | linear_momentum_rel_drift=0.003866; angular_momentum_rel_drift=4.209e-04; kinetic_energy_rel_change_final=-0.001055 | linear_momentum_rel_drift=0; angular_momentum_rel_drift=0 | linear_momentum_rel_drift=1.000e-06; angular_momentum_rel_drift=0.001; velocities_derived=False |
| a1b_conservation_drifting_v1 | code_verification | A | **PASS** | linear_momentum_rel_drift=4.045e-04; angular_momentum_rel_drift=4.083e-04; kinetic_energy_rel_change_final=3.715e-04; declared_domain_handling=co_moving | linear_momentum_rel_drift=0; angular_momentum_rel_drift=0 | linear_momentum_rel_drift=0.001; angular_momentum_rel_drift=0.01; velocities_derived=False |
| a2_beam_frequency_v1 | solution_verification | A | **NA** |  |  |  |
| a3_wavespeed_v1 | solution_verification | A | **NA** |  |  |  |
| a4_meltfront_v1 | capability | A | **NA** |  |  |  |
| a5_stiffness_v1 | solution_verification | A | **NA** |  |  |  |
| c1_covariance_v1 | code_verification | C | **NA** |  |  |  |
| c2_blobhealth_melt_v1 | capability | C | **NA** |  |  |  |
| c2_blobhealth_shear_v1 | capability | C | **NA** |  |  |  |
| c3_roundtrip_v1 | capability | C | **NA** |  |  |  |
| c4_captured_frequency_v1 | capability | C | **NA** |  |  |  |

### a1a_conservation_inplace_v1
- Angular momentum magnitude trace |L(t)|/|L(0)|: {'t=0.00s': 1.0, 't=2.50s': 0.999925773996118, 't=5.00s': 0.9998127373315172, 't=7.50s': 0.999708291099407, 't=10.00s': 0.9995791821819608}

### a1b_conservation_drifting_v1
- Angular momentum magnitude trace |L(t)|/|L(0)|: {'t=0.00s': 1.0, 't=2.50s': 0.9999265215153658, 't=5.00s': 0.9998121507005564, 't=7.50s': 0.9997077292453902, 't=10.00s': 0.9995917954786772}

### a2_beam_frequency_v1
- No submission directory found. For capability tests this means the system does not support the capability.

### a3_wavespeed_v1
- No submission directory found. For capability tests this means the system does not support the capability.

### a4_meltfront_v1
- No submission directory found. For capability tests this means the system does not support the capability.

### a5_stiffness_v1
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
