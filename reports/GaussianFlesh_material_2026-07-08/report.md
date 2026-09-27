# GaussianBench report

**Tier A**: 1 pass, 0 fail, 6 n/a
**Tier C**: 0 pass, 0 fail, 5 n/a

| scene | class | tier | status | measured | reference | tolerance |
|---|---|---|---|---|---|---|
| a1a_conservation_inplace_v1 | code_verification | A | **NA** |  |  |  |
| a1b_conservation_drifting_v1 | code_verification | A | **NA** |  |  |  |
| a2_beam_frequency_v1 | solution_verification | A | **NA** |  |  |  |
| a3_wavespeed_v1 | solution_verification | A | **NA** |  |  |  |
| a4_meltfront_v1 | capability | A | **NA** |  |  |  |
| a5_stiffness_v1 | solution_verification | A | **NA** |  |  |  |
| c1_covariance_v1 | code_verification | C | **NA** |  |  |  |
| c2_blobhealth_melt_v1 | capability | C | **NA** |  |  |  |
| c2_blobhealth_shear_v1 | capability | C | **NA** |  |  |  |
| c3_roundtrip_v1 | capability | C | **NA** |  |  |  |
| c4_captured_frequency_v1 | capability | C | **NA** |  |  |  |
| m1_material_validation_v1 | model_validation | A | **PASS** | median_rel_error=0.05397; p95_rel_error=0.08729; max_stretch_scored=2.5; max_strain_scored=1.5; n_points=45 | source=Azarniya & Rahimi 2022, ASTM D412 Yeoh fit | median_rel_tolerance=0.15; max_stretch=2.5 |

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

### m1_material_validation_v1
- Uniaxial tension isolates the constitutive law (no clamp, no bending), so this gap is cleanly attributable to the material model.
- Compared against the measured natural-rubber curve up to 2.50x stretch (150% engineering strain).
