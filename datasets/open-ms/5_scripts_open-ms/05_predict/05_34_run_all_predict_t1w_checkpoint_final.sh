#!/usr/bin/env bash
# Re-predict the t1w suite's EXISTING checkpoint_best RUN_IDs at checkpoint_final.pth,
# for the checkpoint_best vs checkpoint_final comparison (see
# 06_evaluate/configs/open-ms_t1w_01_results_*_ckpt.yaml). 05_32 (auglabAug_v26_6_2) is contrast-agnostic -- it takes DATASET_ID from whatever env was sourced, here env_t1w.sh (71) via the explicit override below.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env_t1w.sh"
cd "${PROJECT_ROOT}"

METHOD_SCRIPTS=(
    "${HERE}/05_11_predict_t1w_baseline.sh"
    "${HERE}/05_13_predict_t1w_synthseg_noEM.sh"
    "${HERE}/05_14_predict_t1w_synthseg_EM.sh"
    "${HERE}/05_12_predict_t1w_auglab_default.sh"
    "${HERE}/05_19_predict_t1w_srcsm.sh"
    "${HERE}/05_32_predict_auglabAug_v26_6_2_train050_val100.sh"
)
METHOD_RUN_IDS=(
    "open-ms_t1w_baseline_20260708_083441"
    "open-ms_t1w_synthseg_noEM_train100_val000_20260708_083541"
    "open-ms_t1w_synthseg_EM_train100_val000_20260708_083611"
    "open-ms_t1w_auglab_default_20260708_083511"
    "open-ms_t1w_srcsm_20260709_075121"
    "open-ms_t1w_auglabAug_v26_6_2_train050_val100_20260723_194125"
)
export DATASET_ID=71  # 05_32 does not hardcode it (contrast-agnostic wrapper)
export CHECKPOINT=checkpoint_final.pth
source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/run_all_predict_checkpoint_sweep.sh"
