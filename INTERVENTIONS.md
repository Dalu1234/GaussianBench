# GaussianBench Paired Intervention Protocols

These protocols hold the initial geometry and all non-intervened settings
fixed, change one latent physical attribute, and grade the resulting
trajectory divergence against a prediction made from the scene parameters.
They are paired protocols rather than ordinary single-scene scores.

## I1: thermal boundary intervention

- Base scene: `a4_meltfront_v1`
- Runs: hot-floor temperature 370 K and 400 K
- Held fixed: positions, material, initial temperature, boundary type,
  output rate, duration, gravity, and all solver settings
- Readout: for each particle layer, the first time its melt-state fraction
  reaches 50%; the bottom two layers inside the fixed-temperature source band
  are excluded
- Per-run fit: layer height `s = a sqrt(t) + b`
- Prediction: `a_370 / a_400`, computed from the Neumann/Stefan reference
- Pass criteria: ratio relative error below 15% and each layer-crossing fit
  has `R^2 > 0.98`

The original A4 verdicts remain separate. In particular, an intervention
readout never overwrites an A4 result produced by the frozen framewise-front
scorer.

## I2: material-painting intervention

- Base scene: `b1_bimaterial_bar_v1`
- Paintings:
  1. uniform soft (`E_left = E_right = 5e4 Pa`);
  2. soft-left/stiff-right (`5e4 Pa`, `2e5 Pa`);
  3. stiff-left/soft-right (`2e5 Pa`, `5e4 Pa`)
- Held fixed: positions, prescribed end displacement, initial state, output
  rate, duration, gravity, and all solver settings
- Readout: the left half's share of total axial extension, inferred from the
  two frozen B1 core-strain gauges
- Prediction:

  ```
  Delta_left / Delta =
      (L_left / E_left) /
      (L_left / E_left + L_right / E_right)
  ```

- Pass criterion: absolute fraction error below 0.05

## GaussianFlesh reference result

| Protocol/readout | Predicted | Measured | Error | Status |
|---|---:|---:|---:|---|
| I1 `a_370 / a_400` | 0.752348 | 0.754137 | 0.238% | PASS |
| I2 uniform soft | 0.500000 | 0.497987 | 0.002013 | PASS |
| I2 soft-left/stiff-right | 0.800000 | 0.797706 | 0.002294 | PASS |
| I2 stiff-left/soft-right | 0.200000 | 0.199903 | 0.000097 | PASS |

The reproducibility driver and the five frozen intervention scenes are shipped
under `interventions/`. Run
`python interventions/run_counterfactual_interventions.py`; it writes the
full-precision result to
`results/GaussianFlesh_counterfactual_2026-07-24/counterfactual_interventions.json`.
