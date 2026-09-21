#!/usr/bin/env bash
# Source of truth for the brats2024-glioma T1c (contrast-enhanced T1) training runs (TamIA, launched 2026-09-21).
# RUN_IDs are FIXED here (one timestamp) and used by the train orchestrator (04_80), predict job (05_82) and
# eval job (06_31) alike — no hand-copying between them (that mismatch cost time on the T2f port).
# Row: group|name|predict_wrapper|RUN_ID|category|metrics_subdir      (kept at 6 fields: consumers `read` them positionally)
#   group           main = Packs A+B (9 runs); rung4 = ladder rung 4 (Pack C, finishes early); srcsm = ~5 days
#   metrics_subdir  "" = headline (flat under 02_metrics/.../t1c/); "ablations" = ladder-exclusive
# ⚠ This literal timestamp is ALSO copied into configs/brats_t1c_01_results.yaml, configs/brats_combined_4mod_01_results.yaml and
# 06_evaluate/06_33_ladder_summary_t1c.py (YAML/Python can't source this file). 04_80 refuses to re-record an existing pack dir, so a
# relaunch means bumping this — then update those three files too (`grep -rn 20260921_140000` finds them all).
T1C_TS="20260921_140000"
T1C_RUNS=(
  "main|baseline|05_70_predict_t1c_baseline.sh|brats2024-glioma_t1c_baseline_${T1C_TS}|nnUNet|"
  "main|auglab_default|05_71_predict_t1c_auglab_default.sh|brats2024-glioma_t1c_auglab_default_${T1C_TS}|auglab|"
  "main|synthseg_EM|05_72_predict_t1c_synthseg_EM.sh|brats2024-glioma_t1c_synthseg_EM_${T1C_TS}|auglab|"
  "main|synthseg_noEM|05_73_predict_t1c_synthseg_noEM.sh|brats2024-glioma_t1c_synthseg_noEM_${T1C_TS}|auglab|"
  "main|ours_val000|05_75_predict_t1c_auglabAug_v26_6_2_train050_val000_dualval.sh|brats2024-glioma_t1c_auglabAug_v26_6_2_train050_val000_${T1C_TS}|auglab|"
  "main|ours_val100|05_76_predict_t1c_auglabAug_v26_6_2_train050_val100_dualval.sh|brats2024-glioma_t1c_auglabAug_v26_6_2_train050_val100_${T1C_TS}|auglab|"
  "main|rung5_v26_6_2|05_77_predict_t1c_v26_6_2_train050_val100.sh|brats2024-glioma_t1c_v26_6_2_train050_val100_${T1C_TS}|nnUNet|ablations"
  "main|rung2_kmeans|05_78_predict_t1c_baseline_kmeans.sh|brats2024-glioma_t1c_baseline_kmeans_${T1C_TS}|auglab|ablations"
  "main|rung3_kmeans_label_remap|05_79_predict_t1c_baseline_kmeans_label_remap.sh|brats2024-glioma_t1c_baseline_kmeans_label_remap_${T1C_TS}|auglab|ablations"
  "rung4|rung4_voronoi|05_80_predict_t1c_baseline_kmeans_label_remap_voronoi.sh|brats2024-glioma_t1c_baseline_kmeans_label_remap_voronoi_${T1C_TS}|auglab|ablations"
  "srcsm|srcsm|05_74_predict_t1c_srcsm.sh|brats2024-glioma_t1c_srcsm_${T1C_TS}|auglab|"
)
# Training layout: name|train_wrapper|pack. Each pack = one chain of whole-node h100:4 jobs; folds packed concurrently.
#   A (12 folds, 3/GPU) baseline/auglab_default/synthseg_EM/synthseg_noEM    B (12) OURS/rung5/rung2/rung3
#   C (6) rung4 voronoi (3 folds share GPU 3) + srcsm (one GPU per fold: srcsm is ~3x slower/epoch — PACK_GPU_MAP)
T1C_TRAIN=(
  "baseline|04_70_train_t1c_baseline.sh|A"
  "auglab_default|04_71_train_t1c_auglab_default.sh|A"
  "synthseg_EM|04_72_train_t1c_synthseg_EM.sh|A"
  "synthseg_noEM|04_73_train_t1c_synthseg_noEM.sh|A"
  "ours_val000|04_75_train_t1c_auglabAug_v26_6_2_dualval.sh|B"
  "rung5_v26_6_2|04_76_train_t1c_v26_6_2_train050_val100.sh|B"
  "rung2_kmeans|04_77_train_t1c_baseline_kmeans.sh|B"
  "rung3_kmeans_label_remap|04_78_train_t1c_baseline_kmeans_label_remap.sh|B"
  "rung4_voronoi|04_79_train_t1c_baseline_kmeans_label_remap_voronoi.sh|C"
  "srcsm|04_74_train_t1c_srcsm.sh|C"
)
T1C_N_TEST_CASES=70       # held-out test set (4_splits/test_cases.json); audits require exactly this per contrast
T1C_CONTRASTS="t1n t1c t2w t2f"     # TEST contrasts (t1c is now also a training contrast, still evaluated)
