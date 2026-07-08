# ON-Harmony — T2w-trained models, cross-contrast — paired significance (dice)

Generated: 2026-07-06 09:41
Reference (Ours): `on-harmony_T2w_auglabAug_v26_6_2_train025_val100_20260624_191557`

Two-sided Wilcoxon signed-rank on per-case scores (a case's score = mean over its labels and evaluated folds). Unit = case, so the sample size is the number of held-out cases, **not** the fold count. `Δ` = median paired difference (ref − competitor, in Dice pts); W/T/L = cases where ref wins/ties/loses.

## Pooled over all held-out contrasts (Ours vs each competitor)

| competitor | n pairs | Δ median | W/T/L | p (4-fold) | Holm (4f) | p (3-fold) | Holm (3f) |
|---|---|---|---|---|---|---|---|
| on-harmony_T2w_baseline_20260624_191152 | 0 | +nan | 0/0/0 | — | — | — | — |
| on-harmony_T2w_v26_6_2_train050_val100_20260625_154418 | 0 | +nan | 0/0/0 | — | — | — | — |
| on-harmony_T2w_auglab_default_20260625_143115 | 0 | +nan | 0/0/0 | — | — | — | — |
| on-harmony_T2w_synthseg_EM_20260624_191329 | 0 | +nan | 0/0/0 | — | — | — | — |
| on-harmony_T2w_synthseg_noEM_20260624_191418 | 0 | +nan | 0/0/0 | — | — | — | — |

## Per-contrast (raw p; 4-fold vs 3-fold, uncorrected)

| competitor |  |
|---|
| on-harmony_T2w_baseline_20260624_191152 |  |
| on-harmony_T2w_v26_6_2_train050_val100_20260625_154418 |  |
| on-harmony_T2w_auglab_default_20260625_143115 |  |
| on-harmony_T2w_synthseg_EM_20260624_191329 |  |
| on-harmony_T2w_synthseg_noEM_20260624_191418 |  |
