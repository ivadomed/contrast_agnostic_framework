#!/usr/bin/env bash
# Re-predict the t2spir suite's EXISTING checkpoint_best RUN_IDs at checkpoint_final.pth,
# for the checkpoint_best vs checkpoint_final comparison (see
# 06_evaluate/configs/chaos_t2spir_01_results_*_ckpt.yaml). 
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env_t2spir.sh"
cd "${PROJECT_ROOT}"

METHOD_SCRIPTS=(
    "${HERE}/05_21_predict_t2spir_baseline.sh"
    "${HERE}/05_25_predict_t2spir_synthseg_noEM.sh"
    "${HERE}/05_24_predict_t2spir_synthseg_EM.sh"
    "${HERE}/05_23_predict_t2spir_auglab_default.sh"
    "${HERE}/05_33_predict_t2spir_srcsm.sh"
    "${HERE}/05_48_predict_t2spir_auglabAug_v26_6_2_train050_val100.sh"
)
METHOD_RUN_IDS=(
    "chaos_t2spir_baseline_20260620_111146"
    "chaos_t2spir_synthseg_noEM_20260620_112515"
    "chaos_t2spir_synthseg_EM_20260620_112357"
    "chaos_t2spir_auglab_default_20260620_112240"
    "chaos_t2spir_srcsm_20260709_121945"
    "chaos_t2spir_auglabAug_v26_6_2_train050_val100_20260723_194053"
)
export CHECKPOINT=checkpoint_final.pth
source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/run_all_predict_checkpoint_sweep.sh"
