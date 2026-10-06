#!/usr/bin/env bash
# Predict the whole t1wce-trained pansegdata roster (6 headline methods + OURS val100 mirror + ladder rungs 2-5) on t1wce/t2w, all folds.
# RUN_IDs are resolved from the trained run dirs by the shared roster driver (no timestamps here); pins -> roster_run_ids.tsv.
# Usage: bash 05_24_run_all_predict_t1wce.sh            ROSTER_SKIP_MISSING=1 ROSTER_ONLY="baseline srcsm" CHECKPOINT=checkpoint_final.pth ... (see run_all_predict_common.sh)
source "$(dirname "$0")/../00_utils/env.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
METHOD_SCRIPTS=(
    "${HERE}/05_02_predict_t1wce_baseline.sh"
    "${HERE}/05_03_predict_t1wce_auglab_default.sh"
    "${HERE}/05_04_predict_t1wce_synthseg_noEM.sh"
    "${HERE}/05_05_predict_t1wce_synthseg_EM.sh"
    "${HERE}/05_06_predict_t1wce_srcsm.sh"
    "${HERE}/05_07_predict_t1wce_auglabAug_v26_6_2_train050_val000.sh"
    "${HERE}/05_08_predict_t1wce_auglabAug_v26_6_2_train050_val100.sh"
    "${HERE}/05_09_predict_t1wce_baseline_kmeans.sh"
    "${HERE}/05_10_predict_t1wce_baseline_kmeans_label_remap.sh"
    "${HERE}/05_11_predict_t1wce_baseline_kmeans_label_remap_voronoi.sh"
    "${HERE}/05_12_predict_t1wce_v26_6_2_train050_val000.sh"
)
source "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_02_predict/run_all_predict_common.sh"
