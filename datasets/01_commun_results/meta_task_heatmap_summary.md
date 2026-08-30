# Cross-dataset — task-level heatmap (7-method suite, all 6 datasets)

Generated: 2026-08-29 14:53  |  Methods: 7  |  Tasks: brats2024-glioma, chaos, on-harmony, open-ms, healthy-spine-tum, atlas-liver-hcc

Each task column = that dataset's own `all` value (already averaged over its tested contrasts AND its two training modalities — see its `combined_contrasts/01_results_summary.md`). `overall` = equal-weight average across the 6 tasks that have data for that method (a task missing for a given method — e.g. no HD95, or a method not run on that dataset — is excluded, not counted as 0). **Bold** = best per column. `sig. vs ref` = Holm-corrected one-sided (ref better) macroΔ p-value of `auglabAug_v26_6_2_train050_val000 (Ours)` vs that row, equal weight per TASK (blank on the ref's own row); **bold** = p < 0.05.

## Dice ↑

| method | brats2024-glioma | chaos | on-harmony | open-ms | healthy-spine-tum | atlas-liver-hcc | overall | sig. vs ref |
|---|---|---|---|---|---|---|---|---|
| **baseline** | 21.7 | 29.3 | 31.1 | 21.0 | 48.5 | 42.0 | 32.3 | **8.9e-69** |
| **synthseg_noEM** | 8.6 | 59.6 | 66.3 | 4.3 | 50.4 | 19.0 | 34.7 | **1.4e-116** |
| **synthseg_EM** | 43.6 | 83.0 | 67.4 | 33.9 | 80.6 | 47.3 | 59.3 | **1.7e-14** |
| **auglab_default** | 46.7 | 83.3 | 67.4 | 34.3 | 80.9 | 46.9 | 59.9 | **9.9e-12** |
| **srcsm** | 42.1 | 83.9 | 66.6 | 33.0 | — | 41.2 | 53.3 | **1.1e-32** |
| **auglabAug**_**v26_6_2**_train050_val000 **(Ours)** | 46.8 | 85.0 | 68.2 | **38.5** | **82.3** | **49.7** | **61.7** | — |
| **auglabAug**_**v26_6_2**_train050_val100 **(Ours)** | **47.3** | **85.2** | **68.4** | 37.9 | — | 49.1 | 57.6 | 0.7312 |

## HD95 mm ↓

| method | brats2024-glioma | chaos | on-harmony | open-ms | healthy-spine-tum | atlas-liver-hcc | overall | sig. vs ref |
|---|---|---|---|---|---|---|---|---|
| **baseline** | 47.9 | 124.4 | 28.6 | 35.2 | — | 75.5 | 62.3 | **1.4e-29** |
| **synthseg_noEM** | 78.9 | 89.9 | 7.5 | 41.9 | — | 172.0 | 78.0 | **7.7e-114** |
| **synthseg_EM** | 17.4 | 31.8 | 4.7 | 24.7 | — | 70.4 | 29.8 | 1.0000 |
| **auglab_default** | 15.0 | 39.3 | 4.7 | 26.5 | — | 81.6 | 33.4 | **0.0018** |
| **srcsm** | 18.3 | **26.7** | 5.6 | 24.6 | — | 84.1 | 31.9 | **0.0018** |
| **auglabAug**_**v26_6_2**_train050_val000 **(Ours)** | 15.2 | 30.6 | 4.2 | 24.3 | — | 80.0 | 30.8 | — |
| **auglabAug**_**v26_6_2**_train050_val100 **(Ours)** | **14.7** | 28.7 | **4.1** | **24.2** | — | **69.2** | **28.2** | 1.0000 |
