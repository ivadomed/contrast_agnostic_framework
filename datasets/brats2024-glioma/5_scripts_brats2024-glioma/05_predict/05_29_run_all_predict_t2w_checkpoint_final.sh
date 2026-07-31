#!/usr/bin/env bash
# Re-predict the t2w 6-method suite's EXISTING checkpoint_best RUN_IDs at
# checkpoint_final.pth, for the checkpoint_best vs checkpoint_final comparison
# (see 06_evaluate/configs/brats_t2w_01_results_*_ckpt.yaml). 
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env_t2w.sh"
cd "${PROJECT_ROOT}"

METHOD_SCRIPTS=(
    "${HERE}/05_09_predict_t2w_baseline.sh"
    "${HERE}/05_12_predict_t2w_synthseg_noEM.sh"
    "${HERE}/05_11_predict_t2w_synthseg_EM.sh"
    "${HERE}/05_10_predict_t2w_auglab_default.sh"
    "${HERE}/05_19_predict_t2w_srcsm.sh"
    "${HERE}/05_23_predict_t2w_auglabAug_v26_6_2_train050_val100_dualval.sh"
)
METHOD_RUN_IDS=(
    "brats2024-glioma_t2w_baseline_20260620_125115"
    "brats2024-glioma_t2w_synthseg_noEM_20260620_125442"
    "brats2024-glioma_t2w_synthseg_EM_20260620_125354"
    "brats2024-glioma_t2w_auglab_default_20260620_125306"
    "brats2024-glioma_t2w_srcsm_20260709_122045"
    "brats2024-glioma_t2w_auglabAug_v26_6_2_train050_val100_20260725_113540"
)
export CHECKPOINT=checkpoint_final.pth
source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/run_all_predict_checkpoint_sweep.sh"
