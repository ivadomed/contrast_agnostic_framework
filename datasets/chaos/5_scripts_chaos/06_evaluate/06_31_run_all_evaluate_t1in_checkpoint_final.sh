#!/usr/bin/env bash
# Evaluate the t1in suite's checkpoint_final predictions (produced by
# 05_predict/05_49_run_all_predict_t1in_checkpoint_final.sh), for the checkpoint_best vs checkpoint_final comparison
# (see 06_evaluate/configs/chaos_t1in_03_results_*_ckpt.yaml). t1in is this dataset's default DATASET_ID (60), no override needed.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

EVAL_SCRIPT="${HERE}/06_01_evaluate_run.sh"
EVAL_ARGS=(
    "chaos_t1in_srcsm_20260710_011817"
    "chaos_t1in_auglabAug_v26_6_2_train050_val100_20260616_112420"
)
export CKPT_TAG=final
source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/run_all_evaluate_checkpoint_sweep.sh"
