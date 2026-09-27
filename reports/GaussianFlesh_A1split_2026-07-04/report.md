# GaussianBench report

**Tier A**: 0 pass, 2 fail, 4 n/a
**Tier C**: 0 pass, 0 fail, 5 n/a

| scene | class | tier | status | measured | reference | tolerance |
|---|---|---|---|---|---|---|
| a1a_conservation_inplace_v1 | code_verification | A | **FAIL** | linear_momentum_rel_drift=0.00291; angular_momentum_rel_drift=0.1641; kinetic_energy_rel_change_final=-0.2989 | linear_momentum_rel_drift=0; angular_momentum_rel_drift=0 | linear_momentum_rel_drift=1.000e-06; angular_momentum_rel_drift=0.001; velocities_derived=False |
| a1b_conservation_drifting_v1 | code_verification | A | **FAIL** | linear_momentum_rel_drift=0.001359; angular_momentum_rel_drift=2.195; kinetic_energy_rel_change_final=-0.01122; declared_domain_handling=fixed | linear_momentum_rel_drift=0; angular_momentum_rel_drift=0 | linear_momentum_rel_drift=0.001; angular_momentum_rel_drift=0.01; velocities_derived=False |
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
- Angular momentum magnitude trace |L(t)|/|L(0)|: {'t=0.00s': 1.0, 't=2.50s': 0.8379590089517613, 't=5.00s': 0.8377832075876982, 't=7.50s': 0.8375913165596468, 't=10.00s': 0.8374007610450991}

### a1b_conservation_drifting_v1
- Angular momentum magnitude trace |L(t)|/|L(0)|: {'t=0.00s': 1.0, 't=2.50s': 0.3491382045425679, 't=5.00s': 0.6677547023279697, 't=7.50s': 1.5062803341693318, 't=10.00s': 3.1426859849541056}

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
