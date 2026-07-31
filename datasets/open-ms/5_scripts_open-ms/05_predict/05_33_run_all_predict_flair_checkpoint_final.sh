#!/usr/bin/env bash
# Re-predict the flair suite's EXISTING checkpoint_best RUN_IDs at checkpoint_final.pth,
# for the checkpoint_best vs checkpoint_final comparison (see
# 06_evaluate/configs/open-ms_flair_01_results_*_ckpt.yaml). 
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

METHOD_SCRIPTS=(
    "${HERE}/05_04_predict_baseline.sh"
    "${HERE}/05_06_predict_synthseg_noEM.sh"
    "${HERE}/05_02_predict_synthseg_EM.sh"
    "${HERE}/05_05_predict_auglab_default.sh"
    "${HERE}/05_18_predict_srcsm.sh"
    "${HERE}/05_32_predict_auglabAug_v26_6_2_train050_val100.sh"
)
METHOD_RUN_IDS=(
    "open-ms_flair_baseline_20260706_061243"
    "open-ms_flair_synthseg_noEM_train100_val000_20260706_061243"
    "open-ms_flair_synthseg_EM_train100_val000_20260706_061243"
    "open-ms_flair_auglab_default_20260706_061243"
    "open-ms_flair_srcsm_20260709_072043"
    "open-ms_flair_auglabAug_v26_6_2_train050_val100_20260716_095413"
)
export CHECKPOINT=checkpoint_final.pth
source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/run_all_predict_checkpoint_sweep.sh"
