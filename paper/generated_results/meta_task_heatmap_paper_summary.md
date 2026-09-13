# Cross-dataset — task-level heatmap (7-method suite, paper's 6 tasks)

Generated: 2026-09-13 13:58  |  Methods: 7  |  Tasks: brats2024-glioma, chaos, on-harmony, open-ms, toothfairy2, ispy2

Each task column = that dataset's own `all` value (already averaged over its tested contrasts AND its two training modalities — see its `combined_contrasts/01_results_summary.md`). `overall` = equal-weight average across the 6 tasks that have data for that method (a task missing for a given method — e.g. no HD95, or a method not run on that dataset — is excluded, not counted as 0). **Bold** = best per column. `sig. vs ref` = Holm-corrected one-sided (ref better) macroΔ p-value of `auglabAug_v26_6_2_train050_val000 (Ours)` vs that row, equal weight per TASK (blank on the ref's own row); **bold** = p < 0.05.

## Dice ↑

| method | brats2024-glioma | chaos | on-harmony | open-ms | toothfairy2 | ispy2 | overall | sig. vs ref |
|---|---|---|---|---|---|---|---|---|
| **baseline** | 21.7 | 29.3 | 31.1 | 21.0 | 59.2 | 35.2 | 32.9 | **9.7e-46** |
| **synthseg_noEM** | 8.6 | 59.6 | 66.3 | 4.3 | 52.9 | 11.4 | 33.8 | **2.0e-95** |
| **synthseg_EM** | 43.6 | 83.0 | 67.4 | 33.9 | 76.2 | 30.3 | 55.7 | **2.3e-35** |
| **auglab_default** | 46.7 | 83.3 | 67.4 | 34.3 | **76.4** | **38.6** | 57.8 | **0.0491** |
| **srcsm** | 42.1 | 83.9 | 66.6 | 33.0 | 61.9 | 11.5 | 49.8 | **1.1e-90** |
| **auglabAug**_**v26_6_2**_train050_val000 **(Ours)** | 46.8 | 85.0 | 68.2 | **38.5** | 76.3 | 36.9 | 58.6 | — |
| **auglabAug**_**v26_6_2**_train050_val100 **(Ours)** | **47.3** | **85.2** | **68.4** | 37.9 | — | — | **59.7** | 0.4947 |

## HD95 mm ↓

| method | brats2024-glioma | chaos | on-harmony | open-ms | toothfairy2 | ispy2 | overall | sig. vs ref |
|---|---|---|---|---|---|---|---|---|
| **baseline** | 47.9 | 124.4 | 28.6 | 35.2 | **20.8** | 135.6 | 65.4 | **2.2e-27** |
| **synthseg_noEM** | 78.9 | 89.9 | 7.5 | 41.9 | 33.3 | 168.7 | 70.0 | **1.4e-91** |
| **synthseg_EM** | 17.4 | 31.8 | 4.7 | 24.7 | 24.8 | 147.5 | 41.8 | **1.6e-16** |
| **auglab_default** | 15.0 | 39.3 | 4.7 | 26.5 | 23.1 | **132.2** | 40.1 | 1.0000 |
| **srcsm** | 18.3 | **26.7** | 5.6 | 24.6 | 29.5 | 168.6 | 45.6 | **1.2e-32** |
| **auglabAug**_**v26_6_2**_train050_val000 **(Ours)** | 15.2 | 30.6 | 4.2 | 24.3 | 24.2 | 139.8 | 39.7 | — |
| **auglabAug**_**v26_6_2**_train050_val100 **(Ours)** | **14.7** | 28.7 | **4.1** | **24.2** | — | — | **17.9** | 1.0000 |
