#!/usr/bin/env bash
# Predict the whole flair-trained isles2022 roster (6 headline methods + OURS val100 mirror + ladder rungs 2-5) on dwi/adc/flair, all folds.
# RUN_IDs are resolved from the trained run dirs by the shared roster driver (no timestamps here); pins -> roster_run_ids.tsv.
# Usage: bash 05_25_run_all_predict_flair.sh            ROSTER_SKIP_MISSING=1 ROSTER_ONLY="baseline srcsm" CHECKPOINT=checkpoint_final.pth ... (see run_all_predict_common.sh)
source "$(dirname "$0")/../00_utils/env_flair.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
METHOD_SCRIPTS=(
    "${HERE}/05_13_predict_flair_baseline.sh"
    "${HERE}/05_14_predict_flair_auglab_default.sh"
    "${HERE}/05_15_predict_flair_synthseg_noEM.sh"
    "${HERE}/05_16_predict_flair_synthseg_EM.sh"
    "${HERE}/05_17_predict_flair_srcsm.sh"
    "${HERE}/05_18_predict_flair_auglabAug_v26_6_2_train050_val000.sh"
    "${HERE}/05_19_predict_flair_auglabAug_v26_6_2_train050_val100.sh"
    "${HERE}/05_20_predict_flair_baseline_kmeans.sh"
    "${HERE}/05_21_predict_flair_baseline_kmeans_label_remap.sh"
    "${HERE}/05_22_predict_flair_baseline_kmeans_label_remap_voronoi.sh"
    "${HERE}/05_23_predict_flair_v26_6_2_train050_val100.sh"
)
source "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_02_predict/run_all_predict_common.sh"
