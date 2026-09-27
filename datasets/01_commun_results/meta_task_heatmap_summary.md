# Cross-dataset — task-level heatmap (7-method suite, all 8 datasets)

Generated: 2026-09-21 10:21  |  Methods: 7  |  Tasks: brats2024-glioma, chaos, on-harmony, open-ms, toothfairy2, ispy2, healthy-spine-tum, totalseg-pelvic

Each task column = that dataset's own `all` value (already averaged over its tested contrasts AND its two training modalities — see its `combined_contrasts/01_results_summary.md`). `overall` = equal-weight average across the 8 tasks that have data for that method (a task missing for a given method — e.g. no HD95, or a method not run on that dataset — is excluded, not counted as 0). **Bold** = best per column. `sig. vs ref` = Holm-corrected one-sided (ref better) macroΔ p-value of `auglabAug_v26_6_2_train050_val000 (Ours)` vs that row, equal weight per TASK (blank on the ref's own row); **bold** = p < 0.05.

## Dice ↑

| method | brats2024-glioma | chaos | on-harmony | open-ms | toothfairy2 | ispy2 | healthy-spine-tum | totalseg-pelvic | overall | sig. vs ref |
|---|---|---|---|---|---|---|---|---|---|---|
| **baseline** | 21.7 | 29.3 | 31.1 | 21.0 | 59.2 | 39.4 | 48.5 | 29.8 | 35.0 | **8.5e-97** |
| **synthseg_noEM** | 8.6 | 59.6 | 66.3 | 4.3 | 53.6 | 14.0 | 50.4 | 15.0 | 34.0 | **1.1e-100** |
| **synthseg_EM** | 43.6 | 83.0 | 67.4 | 33.9 | 76.3 | 36.8 | 80.6 | 62.5 | 60.5 | **2.1e-48** |
| **auglab_default** | 46.7 | 83.3 | 67.4 | 34.3 | 78.3 | **43.8** | 80.9 | 64.3 | 62.4 | **6.1e-08** |
| **srcsm** | 42.1 | 83.9 | 66.6 | 33.0 | **79.0** | 14.0 | — | 65.3 | 54.8 | **2.5e-59** |
| **auglabAug**_**v26_6_2**_train050_val000 **(Ours)** | 46.8 | 85.0 | 68.2 | **38.5** | 77.7 | 43.0 | **82.3** | **65.6** | **63.4** | — |
| **auglabAug**_**v26_6_2**_train050_val100 **(Ours)** | **47.3** | **85.2** | **68.4** | 37.9 | — | — | — | — | 59.7 | 0.8789 |

## HD95 mm ↓

| method | brats2024-glioma | chaos | on-harmony | open-ms | toothfairy2 | ispy2 | healthy-spine-tum | totalseg-pelvic | overall | sig. vs ref |
|---|---|---|---|---|---|---|---|---|---|---|
| **baseline** | 47.9 | 124.4 | 28.6 | 35.2 | 21.7 | 88.4 | — | 149.3 | 70.8 | **4.4e-45** |
| **synthseg_noEM** | 78.9 | 89.9 | 7.5 | 41.9 | 32.9 | 129.6 | — | 261.8 | 91.8 | **7.0e-37** |
| **synthseg_EM** | 17.4 | 31.8 | 4.7 | 24.7 | 24.9 | 106.8 | — | 34.0 | 34.9 | **1.3e-13** |
| **auglab_default** | 15.0 | 39.3 | 4.7 | 26.5 | 22.1 | **83.7** | — | 35.6 | 32.4 | **8.6e-05** |
| **srcsm** | 18.3 | **26.7** | 5.6 | 24.6 | **19.3** | 127.1 | — | 33.9 | 36.5 | **1.3e-20** |
| **auglabAug**_**v26_6_2**_train050_val000 **(Ours)** | 15.2 | 30.6 | 4.2 | 24.3 | 23.5 | 93.9 | — | **29.0** | 31.5 | — |
| **auglabAug**_**v26_6_2**_train050_val100 **(Ours)** | **14.7** | 28.7 | **4.1** | **24.2** | — | — | — | — | **17.9** | 0.8670 |
