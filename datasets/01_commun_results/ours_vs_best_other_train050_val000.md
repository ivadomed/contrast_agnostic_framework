# Ours (auglabAug_v26_6_2, train050_val000) vs best other method — all contrasts (incl. cross-dataset)

Per training set, "best other method" = whichever of {baseline, synthseg_noEM, synthseg_EM, auglab_default, srcsm} has the highest own average Dice over ALL contrasts (in-domain + cross-contrast + cross-dataset, equal weight per contrast — matches the summary table's `all` column). Both Dice and HD95 deltas/significance below are against that SAME method. Δ = Ours − competitor (Dice pts, higher better; HD95 mm, lower better so a NEGATIVE ΔHD95 favors Ours). Significance = Holm-corrected one-sided (Ours better) p from the contrast-stratified sign-flip permutation test in significance_from_config.py, Holm scope = exactly these 5 methods (not the full headline config's run list). **Bold** = p < 0.05.

| dataset | modality | best other method | ΔDice | p (Dice) | ΔHD95 | p (HD95) |
|---|---|---|---|---|---|---|
| brats2024-gli | t1n | synthseg_EM | +0.14 | 0.3357 | +0.09 | 0.5922 |
| brats2024-gli | t2w | auglab_default | +0.22 | 0.2801 | -0.09 | 1.0000 |
| chaos | t1in | srcsm | +1.16 | 0.1036 | +2.05 | 0.7476 |
| chaos | t2spir | auglab_default | +0.79 | **0.0214** | -3.49 | **0.0255** |
| open-ms | flair | srcsm | +1.56 | **0.0098** | +4.92 | 1.0000 |
| open-ms | t1w | synthseg_EM | +0.74 | **0.0167** | +3.33 | 1.0000 |
| on-harmony | T1w | auglab_default | +1.02 | **0.0070** | -0.92 | **0.0079** |
| on-harmony | T2w | synthseg_EM | +0.19 | 0.3038 | -0.01 | 0.4129 |
