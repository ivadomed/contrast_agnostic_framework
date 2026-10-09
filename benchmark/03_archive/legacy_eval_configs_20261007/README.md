# Legacy evaluation configs and their outputs (archived 2026-10-07)

Archived when SRCSM became "SRC + its test-time source matching, as published" across the benchmark (commits 08d010d,
88db8f6): these were the configs that could not be switched because they read superseded data. Nothing here is read by
any live table, ladder or the paper. Moved with plain `mv` (not deleted); paths below are relative to `benchmark/02_tasks/`.

| group | what | why superseded | where it lived |
|---|---|---|---|
| `abdomen_masked_eval/` | amos/sliver07 `*_{t1in,t2spir}_00_comparison.yaml` + `01_comparison.md` | score chaos models on FULL-FOV amos/sliver07 volumes with the CHAOS slab applied as a mask at scoring time; the standard since 2026-10-03 is crop-before-predict (`fov_crop/`, read by `chaos_combined_01_results.yaml`) | `abdomen_healthy/{amos,sliver07}/5_scripts_*/06_evaluate/configs/`, outputs in `8_results_*/02_metrics/chaos_model/{t1in,t2spir}/` |
| `brats_cross_dataset_bratsssa/` | `cross_dataset_{t1n,t2w}_01_results.yaml` + `02_cross_dataset_*` tables/heatmaps | pool the archived brats-ssa2024 cohort (`benchmark/03_archive/brats-ssa2024`) | `brain_tumor/brats2024-glioma/5_scripts_*/06_evaluate/configs/`, outputs in `8_results_*/02_metrics/brats2024_glioma_model/{t1n,t2w}/` |
| `ispy2_significance_duke_lr_only/` | `ispy2_{t1wce,t2w}_significance_01.yaml` + `01_results_significance.md` | duke sources are the L-R-only `*_uni` crops; the standard is the A-P cropped `*_uniap` items, pooled in `ispy2_combined_01_results.yaml` (whose table carries the significance column) | `breast_cancer/ispy2/5_scripts_ispy2/06_evaluate/configs/`, outputs in `8_results_ispy2/02_metrics/ispy2_model/{t1wce,t2w}/` |
| `ours_vs_best_other/` | `gen_ours_vs_best_other_tables.py` + `ours_vs_best_other_train050_val{000,100}.md` | built from the cross_dataset_* configs above (brats-ssa2024, mslesseg/ms3seg, masked amos/sliver07); the single "which method wins" answer is the meta task heatmap | `benchmark/00_commun_scripts/00_03_evaluate/`, `benchmark/01_commun_results/` |

All of them name the plain SRC run (no test-time matching) for SRCSM. The shared scripts that consumed them
(`06_02_aggregate_per_organ_from_config.sh` for amos/sliver07, `ispy2/06_03_significance_from_config.sh`) take the
config as an argument and are left in place.
