# Cross-dataset combined significance — OURS (auglabAug_v26_6_2 train050_val000)

Generated: 2026-07-30 16:33  |  metric: dice  |  ref: `auglabAug_v26_6_2_train050_val000`  |  in-domain excluded: True

Experiments (one trained model each): 8. Effect = OOD cross-contrast **macroΔ** (equal weight per held-out contrast, ref − competitor). **sign test** uses win/loss direction only; **Stouffer** combines the per-experiment Wilcoxon p-values with direction (assumes independence); **dataset-clustered** collapses same-dataset experiments to one vote (independence-safe). ✅ = combined p < 0.05.

## vs auglab_default

- Experiments: **8/8 favor ours** (losses 0); mean OOD macroΔ **+2.67**
- **Sign test** (direction only): p = **0.0078**
- **Stouffer combined** (one-sided, ours better): p = **1.2e-09** ✅
- **Dataset-clustered sign test**: 4/4 datasets favor ours (4 total; mixed excluded) → p = **0.12**

| experiment | dataset | #OOD contr | macroΔ | poolΔ mean | Wilcoxon p (2-sided) | favors |
|---|---|---|---|---|---|---|
| chaos t1in | chaos | 3 | +0.85 | +0.50 | 0.31 | ours |
| chaos t2spir | chaos | 3 | +1.31 | +0.29 | 0.83 | ours |
| brats t1n | brats2024-glioma | 3 | +0.85 | +0.85 | 0.15 | ours |
| brats t2w | brats2024-glioma | 3 | +0.49 | +0.49 | 0.0094 | ours |
| on-harmony T1w | on-harmony | 5 | +1.55 | +1.50 | 0.00063 | ours |
| on-harmony T2w | on-harmony | 5 | +0.72 | +0.81 | 0.014 | ours |
| open-ms flair | open-ms | 2 | +13.05 | +13.05 | 3.1e-05 | ours |
| open-ms t1w | open-ms | 2 | +2.53 | +2.53 | 0.12 | ours |

## vs synthseg_EM

- Experiments: **7/8 favor ours** (losses 1); mean OOD macroΔ **+1.30**
- **Sign test** (direction only): p = **0.07**
- **Stouffer combined** (one-sided, ours better): p = **3.4e-13** ✅
- **Dataset-clustered sign test**: 3/3 datasets favor ours (4 total; mixed excluded) → p = **0.25**

| experiment | dataset | #OOD contr | macroΔ | poolΔ mean | Wilcoxon p (2-sided) | favors |
|---|---|---|---|---|---|---|
| chaos t1in | chaos | 3 | +1.25 | +1.09 | 4.8e-05 | ours |
| chaos t2spir | chaos | 3 | +2.18 | +1.79 | 0.02 | ours |
| brats t1n | brats2024-glioma | 3 | +2.69 | +2.69 | 2.3e-09 | ours |
| brats t2w | brats2024-glioma | 3 | +1.33 | +1.33 | 8.8e-05 | ours |
| on-harmony T1w | on-harmony | 5 | +1.06 | +1.06 | 0.011 | ours |
| on-harmony T2w | on-harmony | 5 | -0.28 | -0.32 | 0.45 | competitor |
| open-ms flair | open-ms | 2 | +1.86 | +1.86 | 0.12 | ours |
| open-ms t1w | open-ms | 2 | +0.33 | +0.33 | 0.5 | ours |

## vs srcsm

- Experiments: **8/8 favor ours** (losses 0); mean OOD macroΔ **+4.32**
- **Sign test** (direction only): p = **0.0078**
- **Stouffer combined** (one-sided, ours better): p = **3.8e-24** ✅
- **Dataset-clustered sign test**: 4/4 datasets favor ours (4 total; mixed excluded) → p = **0.12**

| experiment | dataset | #OOD contr | macroΔ | poolΔ mean | Wilcoxon p (2-sided) | favors |
|---|---|---|---|---|---|---|
| chaos t1in | chaos | 3 | +1.94 | +1.51 | 0.06 | ours |
| chaos t2spir | chaos | 3 | +3.65 | +6.29 | 0.00019 | ours |
| brats t1n | brats2024-glioma | 3 | +8.08 | +8.08 | 4.9e-24 | ours |
| brats t2w | brats2024-glioma | 3 | +4.75 | +4.75 | 1.1e-16 | ours |
| on-harmony T1w | on-harmony | 5 | +2.47 | +1.59 | 0.09 | ours |
| on-harmony T2w | on-harmony | 5 | +1.34 | +0.49 | 0.81 | ours |
| open-ms flair | open-ms | 2 | +2.19 | +2.19 | 0.74 | ours |
| open-ms t1w | open-ms | 2 | +10.12 | +10.12 | 3.1e-05 | ours |

