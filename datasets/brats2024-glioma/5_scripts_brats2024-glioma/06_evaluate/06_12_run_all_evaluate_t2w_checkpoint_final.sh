#!/usr/bin/env bash
# Evaluate the t2w 6-method suite's checkpoint_final predictions (produced by
# 05_predict/05_29_run_all_predict_t2w_checkpoint_final.sh), for the checkpoint_best vs checkpoint_final comparison
# (see 06_evaluate/configs/brats_t2w_01_results_*_ckpt.yaml). IMPORTANT: 06_01_evaluate_run.sh defaults DATASET_ID to 051 (t1n) internally -- t2w MUST override to 052 explicitly, confirmed the hard way (silently scored against the wrong ground truth otherwise).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env_t2w.sh"
cd "${PROJECT_ROOT}"

EVAL_SCRIPT="${HERE}/06_01_evaluate_run.sh"
EVAL_ARGS=(
    "brats2024-glioma_t2w_baseline_20260620_125115"
    "brats2024-glioma_t2w_synthseg_noEM_20260620_125442"
    "brats2024-glioma_t2w_synthseg_EM_20260620_125354"
    "brats2024-glioma_t2w_auglab_default_20260620_125306"
    "brats2024-glioma_t2w_srcsm_20260709_122045"
    "brats2024-glioma_t2w_auglabAug_v26_6_2_train050_val100_20260725_113540"
)
export CKPT_TAG=final
export DATASET_ID=052
source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/run_all_evaluate_checkpoint_sweep.sh"
