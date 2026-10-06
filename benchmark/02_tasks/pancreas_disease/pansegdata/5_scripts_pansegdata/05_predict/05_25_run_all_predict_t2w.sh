#!/usr/bin/env bash
# Predict the whole t2w-trained pansegdata roster (6 headline methods + OURS val100 mirror + ladder rungs 2-5) on t1wce/t2w, all folds.
# RUN_IDs are resolved from the trained run dirs by the shared roster driver (no timestamps here); pins -> roster_run_ids.tsv.
# Usage: bash 05_25_run_all_predict_t2w.sh            ROSTER_SKIP_MISSING=1 ROSTER_ONLY="baseline srcsm" CHECKPOINT=checkpoint_final.pth ... (see run_all_predict_common.sh)
source "$(dirname "$0")/../00_utils/env_t2w.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
METHOD_SCRIPTS=(
    "${HERE}/05_13_predict_t2w_baseline.sh"
    "${HERE}/05_14_predict_t2w_auglab_default.sh"
    "${HERE}/05_15_predict_t2w_synthseg_noEM.sh"
    "${HERE}/05_16_predict_t2w_synthseg_EM.sh"
    "${HERE}/05_17_predict_t2w_srcsm.sh"
    "${HERE}/05_18_predict_t2w_auglabAug_v26_6_2_train050_val000.sh"
    "${HERE}/05_19_predict_t2w_auglabAug_v26_6_2_train050_val100.sh"
    "${HERE}/05_20_predict_t2w_baseline_kmeans.sh"
    "${HERE}/05_21_predict_t2w_baseline_kmeans_label_remap.sh"
    "${HERE}/05_22_predict_t2w_baseline_kmeans_label_remap_voronoi.sh"
    "${HERE}/05_23_predict_t2w_v26_6_2_train050_val000.sh"
)
source "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_02_predict/run_all_predict_common.sh"
