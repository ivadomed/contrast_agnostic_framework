# Cross-dataset — task-level heatmap (7-method suite, all 4 datasets)

Generated: 2026-08-01 16:59  |  Methods: 7  |  Tasks: brats2024-glioma, chaos, on-harmony, open-ms

Each task column = that dataset's own `all` value (already averaged over its tested contrasts AND its two training modalities — see its `combined_contrasts/01_results_summary.md`). `overall` = equal-weight average across the 4 tasks. **Bold** = best per column. `sig. vs ref` = Holm-corrected one-sided (ref better) macroΔ p-value of `auglabAug_v26_6_2_train050_val000 (Ours)` vs that row, equal weight per TASK (blank on the ref's own row); **bold** = p < 0.05.

## Dice ↑

| method | brats2024-glioma | chaos | on-harmony | open-ms | overall | sig. vs ref |
|---|---|---|---|---|---|---|
| **baseline** | 21.7 | 34.5 | 31.1 | 21.0 | 27.1 | **1.5e-38** |
| **synthseg_noEM** | 8.6 | 60.5 | 66.3 | 4.3 | 34.9 | **2.3e-49** |
| **synthseg_EM** | 43.6 | 85.6 | 67.4 | 33.9 | 57.6 | **3.9e-14** |
| **auglab_default** | 46.7 | 86.3 | 67.4 | 34.3 | 58.7 | **0.0003** |
| **srcsm** | 42.1 | 85.8 | 66.6 | 33.0 | 56.9 | **3.1e-15** |
| **auglabAug**_**v26_6_2**_train050_val000 **(Ours)** | 46.8 | 87.3 | 68.2 | **38.5** | 60.2 | — |
| **auglabAug**_**v26_6_2**_train050_val100 **(Ours)** | **47.3** | **87.5** | **68.4** | 37.9 | **60.3** | 0.6636 |

## HD95 mm ↓

| method | brats2024-glioma | chaos | on-harmony | open-ms | overall | sig. vs ref |
|---|---|---|---|---|---|---|
| **baseline** | 47.9 | 110.9 | 28.6 | 35.2 | 55.7 | **2.6e-30** |
| **synthseg_noEM** | 78.9 | 83.4 | 7.5 | 41.9 | 52.9 | **2.9e-57** |
| **synthseg_EM** | 17.4 | 25.4 | 4.7 | 24.7 | 18.1 | **0.0021** |
| **auglab_default** | 15.0 | 27.9 | 4.7 | 26.5 | 18.5 | **0.0223** |
| **srcsm** | 18.3 | 23.9 | 5.6 | 24.6 | 18.1 | **5.0e-05** |
| **auglabAug**_**v26_6_2**_train050_val000 **(Ours)** | 15.2 | 24.3 | 4.2 | 24.3 | 17.0 | — |
| **auglabAug**_**v26_6_2**_train050_val100 **(Ours)** | **14.7** | **21.7** | **4.1** | **24.2** | **16.2** | 0.9879 |
