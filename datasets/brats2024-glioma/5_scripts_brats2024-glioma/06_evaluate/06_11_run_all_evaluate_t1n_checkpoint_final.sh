#!/usr/bin/env bash
# Evaluate the t1n 6-method suite's checkpoint_final predictions (produced by
# 05_predict/05_28_run_all_predict_t1n_checkpoint_final.sh), for the checkpoint_best vs checkpoint_final comparison
# (see 06_evaluate/configs/brats_t1n_01_results_*_ckpt.yaml). t1n is this dataset's default DATASET_ID (051), no override needed.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

EVAL_SCRIPT="${HERE}/06_01_evaluate_run.sh"
EVAL_ARGS=(
    "brats2024-glioma_t1n_baseline_20260622_044535"
    "brats2024-glioma_t1n_synthseg_noEM_20260622_044535"
    "brats2024-glioma_t1n_synthseg_EM_20260622_044535"
    "brats2024-glioma_t1n_auglab_default_20260622_044535"
    "brats2024-glioma_t1n_auglabAug_v26_6_2_train050_val100_20260725_113540"
)
export CKPT_TAG=final
source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/run_all_evaluate_checkpoint_sweep.sh"
