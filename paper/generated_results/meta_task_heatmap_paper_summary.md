# Cross-dataset — task-level heatmap (7-method suite, paper's 8 tasks)

Generated: 2026-10-06 12:36  |  Methods: 7  |  Tasks: brats2024-glioma, chaos, on-harmony, open-ms, toothfairy2, ispy2, healthy-spine-tum, totalseg-pelvic

Each task column = that dataset's own `all` value (already averaged over its tested contrasts AND its two training modalities — see its `combined_contrasts/01_results_summary.md`). `overall` = equal-weight average across the 8 tasks that have data for that method (a task missing for a given method — e.g. no HD95, or a method not run on that dataset — is excluded, not counted as 0). **Bold** = best per column. `sig. vs ref` = Holm-corrected one-sided (ref better) macroΔ p-value of `auglabAug_v26_6_2_train050_val000 (Ours)` vs that row, equal weight per TASK (blank on the ref's own row); **bold** = p < 0.05. **Per task:** ★ = Ours significantly better than that row *within that task* (Holm over the competitors, one-sided p<0.05), ▼ = significantly worse, none = n.s.; last row counts the ★. The val100 mirror is listed last for reference only — excluded from bold, `overall` ranking, significance and counts.

## Dice ↑

| method | brats2024-glioma | chaos | on-harmony | open-ms | toothfairy2 | ispy2 | healthy-spine-tum | totalseg-pelvic | overall | sig. vs ref |
|---|---|---|---|---|---|---|---|---|---|---|
| **baseline** | 23.4 ★ | 29.5 ★ | 34.5 ★ | 21.0 ★ | 59.2 ★ | 35.0 ★ | 48.5 ★ | 29.8 ★ | 35.1 | **1.9e-75** |
| **synthseg_noEM** | 8.8 ★ | 60.9 ★ | 64.1 ★ | 4.3 ★ | 53.6 ★ | 16.5 ★ | 47.5 ★ | 15.0 ★ | 33.8 | **1.5e-86** |
| **synthseg_EM** | 43.1 ★ | 84.0 ★ | 66.1 | 33.9 ★ | 76.3 ★ | 39.6 ★ | 80.6 ★ | 62.5 ★ | 60.8 | **9.5e-36** |
| **auglab_default** | 44.8 ★ | 84.6 ★ | 66.0 ★ | 34.3 ★ | 78.3 ▼ | 40.1 ★ | 80.9 ★ | 64.3 | 61.7 | **1.1e-13** |
| **srcsm** | 41.5 ★ | 84.6 ★ | **66.8** | 33.0 ★ | **79.0** ▼ | 16.4 ★ | — | 65.3 | 55.2 | **3.5e-84** |
| **auglabAug**_**v26_6_2**_train050_val000 **(Ours)** | **45.9** | **85.9** | 66.7 | **38.5** | 77.7 | **41.2** | **82.3** | **65.6** | **63.0** | — |
| **auglabAug**_**v26_6_2**_train050_val100 **(Ours)** | 46.6 | 86.0 | 66.5 | 37.9 | — | — | — | — | 59.3 | (ref only) |
| **Ours sig. wins** | 5/5 | 5/5 | 3/5 | 5/5 | 3/5 | 5/5 | 4/5 | 3/5 | | |

## HD95 mm ↓

| method | brats2024-glioma | chaos | on-harmony | open-ms | toothfairy2 | ispy2 | healthy-spine-tum | totalseg-pelvic | overall | sig. vs ref |
|---|---|---|---|---|---|---|---|---|---|---|
| **baseline** | 44.8 ★ | 122.1 ★ | 30.3 ★ | 35.2 ★ | 21.7 ▼ | 58.5 | 49.9 ★ | 149.3 ★ | 64.0 | **1.6e-47** |
| **synthseg_noEM** | 78.0 ★ | 80.6 ★ | 8.8 ★ | 41.9 ★ | 32.9 ★ | 101.1 ★ | 111.8 ★ | 261.8 ★ | 89.6 | **1.6e-58** |
| **synthseg_EM** | 17.9 ★ | 23.7 ★ | 5.3 | 24.7 | 24.9 ★ | 60.9 ★ | 4.0 ★ | 34.0 | 24.4 | **2.7e-06** |
| **auglab_default** | 16.1 ★ | 27.8 ★ | 5.4 | 26.5 ★ | 22.1 ▼ | **57.1** | 4.7 ★ | 35.6 | 24.4 | **0.0002** |
| **srcsm** | 17.5 ★ | 24.4 | 5.2 | 24.6 | **19.3** ▼ | 91.8 ★ | — | 33.9 | 31.0 | **1.9e-32** |
| **auglabAug**_**v26_6_2**_train050_val000 **(Ours)** | **15.6** | **22.7** | **5.2** | **24.3** | 23.5 | 57.7 | **3.5** | **29.0** | **22.7** | — |
| **auglabAug**_**v26_6_2**_train050_val100 **(Ours)** | 15.1 | 20.6 | 4.7 | 24.2 | — | — | — | — | 16.1 | (ref only) |
| **Ours sig. wins** | 5/5 | 4/5 | 2/5 | 3/5 | 2/5 | 3/5 | 4/5 | 2/5 | | |
