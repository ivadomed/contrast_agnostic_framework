#!/usr/bin/env bash
# Evaluate the t1w suite's checkpoint_final predictions (produced by
# 05_predict/05_34_run_all_predict_t1w_checkpoint_final.sh), for the checkpoint_best vs checkpoint_final comparison
# (see 06_evaluate/configs/open-ms_t1w_01_results_*_ckpt.yaml). CATEGORY mandatory (see flair's launcher note). DATASET_ID=70/71 is documented as not affecting scores here (GT masks are byte-identical across open-ms's flair/t1w datasets) but set explicitly for clarity anyway.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env_t1w.sh"
cd "${PROJECT_ROOT}"

EVAL_SCRIPT="${HERE}/06_01_evaluate_run.sh"
EVAL_ARGS=(
    "open-ms_t1w_baseline_20260708_083441 nnUNet"
    "open-ms_t1w_synthseg_noEM_train100_val000_20260708_083541 auglab"
    "open-ms_t1w_synthseg_EM_train100_val000_20260708_083611 auglab"
    "open-ms_t1w_auglab_default_20260708_083511 auglab"
    "open-ms_t1w_srcsm_20260709_075121 auglab"
    "open-ms_t1w_auglabAug_v26_6_2_train050_val100_20260723_194125 auglab"
)
export CKPT_TAG=final
export DATASET_ID=71
source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/run_all_evaluate_checkpoint_sweep.sh"
