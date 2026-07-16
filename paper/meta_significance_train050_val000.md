# Cross-dataset combined significance — OURS (auglabAug_v26_6_2 train050_val000)

Generated: 2026-07-16 04:07  |  metric: dice  |  ref: `auglabAug_v26_6_2_train050_val000`  |  in-domain excluded: True

Experiments (one trained model each): 6. Effect = OOD cross-contrast **macroΔ** (equal weight per held-out contrast, ref − competitor). **sign test** uses win/loss direction only; **Stouffer** combines the per-experiment Wilcoxon p-values with direction (assumes independence); **dataset-clustered** collapses same-dataset experiments to one vote (independence-safe). ✅ = combined p < 0.05.

## vs auglab_default

- Experiments: **6/6 favor ours** (losses 0); mean OOD macroΔ **+3.22**
- **Sign test** (direction only): p = **0.031**
- **Stouffer combined** (one-sided, ours better): p = **4.6e-06** ✅
- **Dataset-clustered sign test**: 4/4 datasets favor ours (4 total; mixed excluded) → p = **0.12**

| experiment | dataset | #OOD contr | macroΔ | poolΔ mean | Wilcoxon p (2-sided) | favors |
|---|---|---|---|---|---|---|
| chaos t1in | chaos | 3 | +0.85 | +0.50 | 0.31 | ours |
| chaos t2spir | chaos | 3 | +1.31 | +0.29 | 0.83 | ours |
| brats t1n | brats2024-glioma | 3 | +0.85 | +0.85 | 0.15 | ours |
| on-harmony T2w | on-harmony | 5 | +0.72 | +0.81 | 0.014 | ours |
| open-ms flair | open-ms | 2 | +13.05 | +13.05 | 3.1e-05 | ours |
| open-ms t1w | open-ms | 2 | +2.53 | +2.53 | 0.12 | ours |

## vs synthseg_EM

- Experiments: **5/6 favor ours** (losses 1); mean OOD macroΔ **+1.34**
- **Sign test** (direction only): p = **0.22**
- **Stouffer combined** (one-sided, ours better): p = **7.7e-09** ✅
- **Dataset-clustered sign test**: 3/4 datasets favor ours (4 total; mixed excluded) → p = **0.62**

| experiment | dataset | #OOD contr | macroΔ | poolΔ mean | Wilcoxon p (2-sided) | favors |
|---|---|---|---|---|---|---|
| chaos t1in | chaos | 3 | +1.25 | +1.09 | 4.8e-05 | ours |
| chaos t2spir | chaos | 3 | +2.18 | +1.79 | 0.02 | ours |
| brats t1n | brats2024-glioma | 3 | +2.69 | +2.69 | 2.3e-09 | ours |
| on-harmony T2w | on-harmony | 5 | -0.28 | -0.32 | 0.45 | competitor |
| open-ms flair | open-ms | 2 | +1.86 | +1.86 | 0.12 | ours |
| open-ms t1w | open-ms | 2 | +0.33 | +0.33 | 0.5 | ours |

## vs srcsm

- Experiments: **5/5 favor ours** (losses 0); mean OOD macroΔ **+3.85**
- **Sign test** (direction only): p = **0.062**
- **Stouffer combined** (one-sided, ours better): p = **1.9e-06** ✅
- **Dataset-clustered sign test**: 3/3 datasets favor ours (3 total; mixed excluded) → p = **0.25**

| experiment | dataset | #OOD contr | macroΔ | poolΔ mean | Wilcoxon p (2-sided) | favors |
|---|---|---|---|---|---|---|
| chaos t1in | chaos | 3 | +1.94 | +1.51 | 0.06 | ours |
| chaos t2spir | chaos | 3 | +3.65 | +6.29 | 0.00019 | ours |
| on-harmony T2w | on-harmony | 5 | +1.34 | +0.49 | 0.81 | ours |
| open-ms flair | open-ms | 2 | +2.19 | +2.19 | 0.74 | ours |
| open-ms t1w | open-ms | 2 | +10.12 | +10.12 | 3.1e-05 | ours |

