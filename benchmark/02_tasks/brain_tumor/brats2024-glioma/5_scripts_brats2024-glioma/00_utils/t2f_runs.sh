#!/usr/bin/env bash
# Source of truth for the brats2024-glioma T2f/FLAIR training runs (TamIA, 2026-09-17).
# Sourced by 05_predict/05_62_t2f_predict_job.sh and 06_evaluate/06_21_t2f_eval_job.sh so the
# RUN_IDs are specified exactly once. Row: group|name|predict_wrapper|RUN_ID|category|metrics_subdir
#   group           "main" = finished 2026-09-19; "srcsm" = slower, queued behind its training chain
#   metrics_subdir  "" = headline (flat under 02_metrics/.../t2f/); "ablations" = ladder-exclusive
# RUN_ID timestamps differ ON PURPOSE: 20260917_113412 = Packs A/B (re-launched after the
# nnUNet_results path fix); 20260917_094019 = Pack C (never affected, kept running).
T2F_RUNS=(
  "main|baseline|05_50_predict_t2f_baseline.sh|brats2024-glioma_t2f_baseline_20260917_113412|nnUNet|"
  "main|auglab_default|05_51_predict_t2f_auglab_default.sh|brats2024-glioma_t2f_auglab_default_20260917_113412|auglab|"
  "main|synthseg_EM|05_52_predict_t2f_synthseg_EM.sh|brats2024-glioma_t2f_synthseg_EM_20260917_113412|auglab|"
  "main|synthseg_noEM|05_53_predict_t2f_synthseg_noEM.sh|brats2024-glioma_t2f_synthseg_noEM_20260917_113412|auglab|"
  "main|ours_val000|05_55_predict_t2f_auglabAug_v26_6_2_train050_val000_dualval.sh|brats2024-glioma_t2f_auglabAug_v26_6_2_train050_val000_20260917_113412|auglab|"
  "main|ours_val100|05_56_predict_t2f_auglabAug_v26_6_2_train050_val100_dualval.sh|brats2024-glioma_t2f_auglabAug_v26_6_2_train050_val100_20260917_113412|auglab|"
  "main|rung5_v26_6_2|05_57_predict_t2f_v26_6_2_train050_val100.sh|brats2024-glioma_t2f_v26_6_2_train050_val100_20260917_113412|nnUNet|ablations"
  "main|rung2_kmeans|05_58_predict_t2f_baseline_kmeans.sh|brats2024-glioma_t2f_baseline_kmeans_20260917_113412|auglab|ablations"
  "main|rung3_kmeans_label_remap|05_59_predict_t2f_baseline_kmeans_label_remap.sh|brats2024-glioma_t2f_baseline_kmeans_label_remap_20260917_113412|auglab|ablations"
  "main|rung4_voronoi|05_60_predict_t2f_baseline_kmeans_label_remap_voronoi.sh|brats2024-glioma_t2f_baseline_kmeans_label_remap_voronoi_20260917_094019|auglab|ablations"
  "srcsm|srcsm|05_54_predict_t2f_srcsm.sh|brats2024-glioma_t2f_srcsm_20260917_094019|auglab|"
)
T2F_N_TEST_CASES=70       # held-out test set (4_splits/test_cases.json); audits require exactly this per contrast
T2F_CONTRASTS="t1n t1c t2w t2f"
