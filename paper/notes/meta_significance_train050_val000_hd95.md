# Cross-dataset combined significance — OURS (auglabAug_v26_6_2 train050_val000)

Generated: 2026-07-30 16:33  |  metric: hd95  |  ref: `auglabAug_v26_6_2_train050_val000`  |  in-domain excluded: True

Experiments (one trained model each): 8. Effect = OOD cross-contrast **macroΔ** (equal weight per held-out contrast, ref − competitor). **sign test** uses win/loss direction only; **Stouffer** combines the per-experiment Wilcoxon p-values with direction (assumes independence); **dataset-clustered** collapses same-dataset experiments to one vote (independence-safe). ✅ = combined p < 0.05.

## vs auglab_default

- Experiments: **5/8 favor ours** (losses 3); mean OOD macroΔ **-1.97**
- **Sign test** (direction only): p = **0.73**
- **Stouffer combined** (one-sided, ours better): p = **1.5e-07** ✅
- **Dataset-clustered sign test**: 2/3 datasets favor ours (4 total; mixed excluded) → p = **1**

| experiment | dataset | #OOD contr | macroΔ | poolΔ mean | Wilcoxon p (2-sided) | favors |
|---|---|---|---|---|---|---|
| chaos t1in | chaos | 3 | -8.50 | -3.01 | 0.54 | ours |
| chaos t2spir | chaos | 3 | +1.76 | +0.88 | 0.8 | competitor |
| brats t1n | brats2024-glioma | 3 | +0.14 | +0.14 | 0.86 | competitor |
| brats t2w | brats2024-glioma | 3 | +0.00 | +0.00 | 0.38 | competitor |
| on-harmony T1w | on-harmony | 5 | -1.13 | -0.97 | 7.1e-06 | ours |
| on-harmony T2w | on-harmony | 5 | -0.27 | -0.26 | 1.4e-06 | ours |
| open-ms flair | open-ms | 2 | -4.96 | -4.96 | 0.0027 | ours |
| open-ms t1w | open-ms | 2 | -2.81 | -2.81 | 0.0042 | ours |

## vs synthseg_EM

- Experiments: **5/8 favor ours** (losses 3); mean OOD macroΔ **-0.56**
- **Sign test** (direction only): p = **0.73**
- **Stouffer combined** (one-sided, ours better): p = **0.00018** ✅
- **Dataset-clustered sign test**: 1/1 datasets favor ours (4 total; mixed excluded) → p = **1**

| experiment | dataset | #OOD contr | macroΔ | poolΔ mean | Wilcoxon p (2-sided) | favors |
|---|---|---|---|---|---|---|
| chaos t1in | chaos | 3 | -2.69 | -0.87 | 0.17 | ours |
| chaos t2spir | chaos | 3 | +0.54 | -1.11 | 0.73 | competitor |
| brats t1n | brats2024-glioma | 3 | -3.06 | -3.06 | 1.0e-08 | ours |
| brats t2w | brats2024-glioma | 3 | -0.57 | -0.57 | 0.17 | ours |
| on-harmony T1w | on-harmony | 5 | -1.25 | -1.05 | 0.017 | ours |
| on-harmony T2w | on-harmony | 5 | +0.04 | +0.04 | 0.63 | competitor |
| open-ms flair | open-ms | 2 | +3.18 | +3.18 | 0.14 | competitor |
| open-ms t1w | open-ms | 2 | -0.67 | -0.67 | 0.13 | ours |

## vs srcsm

- Experiments: **7/8 favor ours** (losses 1); mean OOD macroΔ **-2.36**
- **Sign test** (direction only): p = **0.07**
- **Stouffer combined** (one-sided, ours better): p = **1.5e-21** ✅
- **Dataset-clustered sign test**: 3/3 datasets favor ours (4 total; mixed excluded) → p = **0.25**

| experiment | dataset | #OOD contr | macroΔ | poolΔ mean | Wilcoxon p (2-sided) | favors |
|---|---|---|---|---|---|---|
| chaos t1in | chaos | 3 | -0.87 | -4.00 | 0.002 | ours |
| chaos t2spir | chaos | 3 | -3.78 | -13.34 | 0.00047 | ours |
| brats t1n | brats2024-glioma | 3 | -7.65 | -7.65 | 3.2e-22 | ours |
| brats t2w | brats2024-glioma | 3 | -2.11 | -2.11 | 1.4e-06 | ours |
| on-harmony T1w | on-harmony | 5 | -2.81 | -2.86 | 0.18 | ours |
| on-harmony T2w | on-harmony | 5 | -0.72 | -0.66 | 0.00028 | ours |
| open-ms flair | open-ms | 2 | +1.98 | +1.98 | 0.98 | competitor |
| open-ms t1w | open-ms | 2 | -2.97 | -2.97 | 0.029 | ours |

