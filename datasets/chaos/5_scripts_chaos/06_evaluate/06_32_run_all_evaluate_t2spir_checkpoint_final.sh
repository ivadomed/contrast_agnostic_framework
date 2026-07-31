#!/usr/bin/env bash
# Evaluate the t2spir suite's checkpoint_final predictions (produced by
# 05_predict/05_50_run_all_predict_t2spir_checkpoint_final.sh), for the checkpoint_best vs checkpoint_final comparison
# (see 06_evaluate/configs/chaos_t2spir_01_results_*_ckpt.yaml). IMPORTANT: 06_01_evaluate_run.sh defaults DATASET_ID to 60 (t1in) internally -- t2spir MUST override to 61 explicitly, confirmed the hard way.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env_t2spir.sh"
cd "${PROJECT_ROOT}"

EVAL_SCRIPT="${HERE}/06_01_evaluate_run.sh"
EVAL_ARGS=(
    "chaos_t2spir_baseline_20260620_111146"
    "chaos_t2spir_synthseg_noEM_20260620_112515"
    "chaos_t2spir_synthseg_EM_20260620_112357"
    "chaos_t2spir_auglab_default_20260620_112240"
    "chaos_t2spir_srcsm_20260709_121945"
    "chaos_t2spir_auglabAug_v26_6_2_train050_val100_20260723_194053"
)
export CKPT_TAG=final
export DATASET_ID=61
source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/run_all_evaluate_checkpoint_sweep.sh"
