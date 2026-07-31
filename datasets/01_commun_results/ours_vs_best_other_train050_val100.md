# Ours (auglabAug_v26_6_2, train050_val100) vs best other method — all contrasts (incl. cross-dataset)

Per training set, "best other method" = whichever of {baseline, synthseg_noEM, synthseg_EM, auglab_default, srcsm} has the highest own average Dice over ALL contrasts (in-domain + cross-contrast + cross-dataset, equal weight per contrast — matches the summary table's `all` column). Both Dice and HD95 deltas/significance below are against that SAME method. Δ = Ours − competitor (Dice pts, higher better; HD95 mm, lower better so a NEGATIVE ΔHD95 favors Ours). Significance = Holm-corrected one-sided (Ours better) p from the contrast-stratified sign-flip permutation test in significance_from_config.py, Holm scope = exactly these 5 methods (not the full headline config's run list). **Bold** = p < 0.05.

| dataset | modality | best other method | ΔDice | p (Dice) | ΔHD95 | p (HD95) |
|---|---|---|---|---|---|---|
| brats2024-gli | t1n | synthseg_EM | -0.41 | 0.8779 | +2.62 | 1.0000 |
| brats2024-gli | t2w | auglab_default | +0.92 | **0.0001** | -0.66 | **0.0396** |
| chaos | t1in | srcsm | +1.07 | 0.1366 | +2.22 | 0.7675 |
| chaos | t2spir | auglab_default | +1.38 | **0.0071** | -6.17 | **0.0000** |
| open-ms | flair | srcsm | +2.59 | **0.0001** | +5.63 | 1.0000 |
| open-ms | t1w | synthseg_EM | -1.03 | 0.9991 | +5.11 | 1.0000 |
| on-harmony | T1w | auglab_default | +1.02 | **0.0070** | -0.92 | **0.0079** |
| on-harmony | T2w | synthseg_EM | +0.60 | 0.1525 | -0.05 | 0.1476 |
