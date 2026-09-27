# GaussianBench report

**Tier A**: 5 pass, 0 fail, 2 n/a
**Tier C**: 4 pass, 0 fail, 1 n/a

| scene | class | tier | status | measured | reference | tolerance |
|---|---|---|---|---|---|---|
| a1a_conservation_inplace_v1 | code_verification | A | **PASS** | linear_momentum_rel_drift=2.833e-16; angular_momentum_rel_drift=4.000e-16; kinetic_energy_rel_change_final=0 | linear_momentum_rel_drift=0; angular_momentum_rel_drift=0 | linear_momentum_rel_drift=1.000e-06; angular_momentum_rel_drift=0.001; velocities_derived=False |
| a1b_conservation_drifting_v1 | code_verification | A | **PASS** | linear_momentum_rel_drift=3.702e-16; angular_momentum_rel_drift=1.020e-14; kinetic_energy_rel_change_final=0; declared_domain_handling=mesh_free | linear_momentum_rel_drift=0; angular_momentum_rel_drift=0 | linear_momentum_rel_drift=0.001; angular_momentum_rel_drift=0.01; velocities_derived=False |
| a2_beam_frequency_v1 | solution_verification | A | **PASS** | fundamental_frequency_hz=1.129; rel_error=1.611e-04; damping_ratio=0.007124; max_tip_deflection_m=0.004813 | fundamental_frequency_hz=1.129 | rel_tolerance=0.1 |
| a3_wavespeed_v1 | solution_verification | A | **PASS** | bar_wave_speed_m_s=22.44; arrival_time_s=0.044; travel_distance_m=0.9875; rel_error=0.00369 | bar_wave_speed_m_s=22.36 | rel_tolerance=0.15 |
| a4_meltfront_v1 | capability | A | **NA** |  |  |  |
| a5_stiffness_v1 | solution_verification | A | **PASS** | lateral_strain=-0.006316; axial_strain=0.02105; rel_error=5.493e-16 | lateral_strain=-0.006316; poisson_ratio=0.3 | rel_tolerance=0.1 |
| c1_covariance_v1 | code_verification | C | **PASS** | median_rel_frobenius_error=0; p95_rel_frobenius_error=3.576e-17; n_particles_scored=3375 | covariance_map=F Sigma_0 F^T; F_target_diag=[1.5, 1.0, 0.8] | median_max=0.001; p95_max=0.01 |
| c2_blobhealth_melt_v1 | capability | C | **NA** |  |  |  |
| c2_blobhealth_shear_v1 | capability | C | **PASS** | min_eigenvalue_ratio=0.382; degenerate_fraction=0; worst_particle_id=0 | healthy_min_eigenvalue_ratio=1; healthy_degenerate_fraction=0 | min_eigenvalue_ratio=0.001; condition_number_max=1.000e+04; degenerate_fraction_max=0.001 |
| c3_roundtrip_v1 | capability | C | **PASS** | psnr_db=100; ssim=1; min_height_frac_during_run=0.78 | psnr_db=inf (identical renders); ssim=1 | psnr_min_db=35; ssim_min=0.97 |
| c4_captured_frequency_v1 | capability | C | **PASS** | fundamental_frequency_hz=0.9074; rel_error=2.917e-06; damping_ratio=0.009516 | fundamental_frequency_hz=0.9074; section_I_over_A_m2=1.317e-04 | rel_tolerance=0.2 |
| m1_material_validation_v1 | model_validation | A | **NA** |  |  |  |

### a1a_conservation_inplace_v1
- No mass column submitted; equal per-particle masses assumed from scene density and geometry volume.
- Angular momentum magnitude trace |L(t)|/|L(0)|: {'t=0.00s': 1.0, 't=2.50s': 1.0, 't=5.00s': 1.0, 't=7.50s': 1.0, 't=10.00s': 1.0}

### a1b_conservation_drifting_v1
- No mass column submitted; equal per-particle masses assumed from scene density and geometry volume.
- Angular momentum magnitude trace |L(t)|/|L(0)|: {'t=0.00s': 1.0, 't=2.50s': 1.0, 't=5.00s': 1.000000000000001, 't=7.50s': 1.0000000000000033, 't=10.00s': 1.0000000000000004}

### a2_beam_frequency_v1
- Damping ratio (report-only, numerical dissipation): zeta = 0.007124.
- Young's modulus used directly in the Euler-Bernoulli formula (uniaxial stress, lateral faces free; no E/(1-nu^2) correction).

### a3_wavespeed_v1
- First motion of the far-end set (mean of 64 particles) at frame 88, t = 0.0440 s.

### a4_meltfront_v1
- No submission directory found. For capability tests this means the system does not support the capability.

### a5_stiffness_v1
- Axial strain at hold: 0.02105 (from measured face displacements over the rest gauge length 0.0950 m).
- Lateral strain measured on core region x in [30%, 70%] of length: eps_y=-0.00632, eps_z=-0.00632.
- Reminder: lateral faces must be traction-free (uniaxial stress). Near-zero contraction usually means your lateral boundaries are clamped; declare boundary handling in meta.json.

### c1_covariance_v1
- Excluded 4625 particles within 2 spacings of a prescribed face; scored 3375.

### c2_blobhealth_melt_v1
- No submission directory found. For capability tests this means the system does not support the capability.

### c2_blobhealth_shear_v1
- Worst particle id 0; the scorer regenerates the eigenvalue history when the full native export is rescored.

### c3_roundtrip_v1
- PSNR reported as 100 dB (images identical, true value infinite).

### c4_captured_frequency_v1
- Reference computed from measured section properties of the irregular asset (see scene provenance); tolerance is wider than A2 accordingly.

### m1_material_validation_v1
- No submission directory found. For capability tests this means the system does not support the capability.
