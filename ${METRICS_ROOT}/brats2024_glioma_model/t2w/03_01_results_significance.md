# BraTS 2024 Glioma T2w — results — paired significance (dice)

Generated: 2026-07-06 09:41
Reference (Ours): `brats2024-glioma_t2w_auglabAug_v26_6_2_train025_val100_20260620_125531`

Two-sided Wilcoxon signed-rank on per-case scores (a case's score = mean over its labels and evaluated folds). Unit = case, so the sample size is the number of held-out cases, **not** the fold count. `Δ` = median paired difference (ref − competitor, in Dice pts); W/T/L = cases where ref wins/ties/loses.

## Pooled over all held-out contrasts (Ours vs each competitor)

| competitor | n pairs | Δ median | W/T/L | p (4-fold) | Holm (4f) | p (3-fold) | Holm (3f) |
|---|---|---|---|---|---|---|---|
| brats2024-glioma_t2w_baseline_20260620_125115 | 0 | +nan | 0/0/0 | — | — | — | — |
| brats2024-glioma_t2w_v26_6_2_train050_val100_20260620_125217 | 0 | +nan | 0/0/0 | — | — | — | — |
| brats2024-glioma_t2w_auglab_default_20260620_125306 | 0 | +nan | 0/0/0 | — | — | — | — |
| brats2024-glioma_t2w_synthseg_EM_20260620_125354 | 0 | +nan | 0/0/0 | — | — | — | — |
| brats2024-glioma_t2w_synthseg_noEM_20260620_125442 | 0 | +nan | 0/0/0 | — | — | — | — |

## Per-contrast (raw p; 4-fold vs 3-fold, uncorrected)

| competitor |  |
|---|
| brats2024-glioma_t2w_baseline_20260620_125115 |  |
| brats2024-glioma_t2w_v26_6_2_train050_val100_20260620_125217 |  |
| brats2024-glioma_t2w_auglab_default_20260620_125306 |  |
| brats2024-glioma_t2w_synthseg_EM_20260620_125354 |  |
| brats2024-glioma_t2w_synthseg_noEM_20260620_125442 |  |
