# Cross-dataset — task-level heatmap (7-method suite, paper's 6 tasks)

Generated: 2026-10-01 07:33  |  Methods: 7  |  Tasks: brats2024-glioma, chaos, on-harmony, open-ms, toothfairy2, ispy2

Each task column = that dataset's own `all` value (already averaged over its tested contrasts AND its two training modalities — see its `combined_contrasts/01_results_summary.md`). `overall` = equal-weight average across the 6 tasks that have data for that method (a task missing for a given method — e.g. no HD95, or a method not run on that dataset — is excluded, not counted as 0). **Bold** = best per column. `sig. vs ref` = Holm-corrected one-sided (ref better) macroΔ p-value of `auglabAug_v26_6_2_train050_val000 (Ours)` vs that row, equal weight per TASK (blank on the ref's own row); **bold** = p < 0.05.

## Dice ↑

| method | brats2024-glioma | chaos | on-harmony | open-ms | toothfairy2 | ispy2 | overall | sig. vs ref |
|---|---|---|---|---|---|---|---|---|
| **baseline** | 23.4 | 29.3 | 35.0 | 21.0 | 59.2 | 35.0 | 33.8 | **2.0e-63** |
| **synthseg_noEM** | 8.8 | 59.6 | 64.8 | 4.3 | 53.6 | 16.5 | 34.6 | **1.4e-67** |
| **synthseg_EM** | 43.1 | 83.0 | 66.3 | 33.9 | 76.3 | 39.6 | 57.0 | **5.6e-26** |
| **auglab_default** | 44.8 | 83.3 | 66.1 | 34.3 | 78.3 | 40.1 | 57.8 | **9.5e-11** |
| **srcsm** | 41.5 | 83.9 | **66.6** | 33.0 | **79.0** | 16.4 | 53.4 | **4.0e-59** |
| **auglabAug**_**v26_6_2**_train050_val000 **(Ours)** | 45.9 | 85.0 | 66.5 | **38.5** | 77.7 | **41.2** | **59.1** | — |
| **auglabAug**_**v26_6_2**_train050_val100 **(Ours)** | **46.6** | **85.2** | 66.6 | 37.9 | — | — | 59.1 | 0.9533 |

## HD95 mm ↓

| method | brats2024-glioma | chaos | on-harmony | open-ms | toothfairy2 | ispy2 | overall | sig. vs ref |
|---|---|---|---|---|---|---|---|---|
| **baseline** | 44.8 | 124.4 | 24.0 | 35.2 | 21.7 | 58.5 | 51.4 | **4.1e-58** |
| **synthseg_noEM** | 78.0 | 89.9 | 6.7 | 41.9 | 32.9 | 101.1 | 58.4 | **5.7e-135** |
| **synthseg_EM** | 17.9 | 31.8 | 4.7 | 24.7 | 24.9 | 60.9 | 27.5 | **2.5e-10** |
| **auglab_default** | 16.1 | 39.3 | 4.7 | 26.5 | 22.1 | **57.1** | 27.6 | **3.1e-16** |
| **srcsm** | 17.5 | **26.7** | 5.3 | 24.6 | **19.3** | 91.8 | 30.9 | **8.7e-34** |
| **auglabAug**_**v26_6_2**_train050_val000 **(Ours)** | 15.6 | 30.6 | 4.3 | 24.3 | 23.5 | 57.7 | 26.0 | — |
| **auglabAug**_**v26_6_2**_train050_val100 **(Ours)** | **15.1** | 28.7 | **4.3** | **24.2** | — | — | **18.1** | 0.8683 |
