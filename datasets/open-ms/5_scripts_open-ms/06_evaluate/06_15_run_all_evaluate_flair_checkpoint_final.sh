#!/usr/bin/env bash
# Evaluate the flair suite's checkpoint_final predictions (produced by
# 05_predict/05_33_run_all_predict_flair_checkpoint_final.sh), for the checkpoint_best vs checkpoint_final comparison
# (see 06_evaluate/configs/open-ms_flair_01_results_*_ckpt.yaml). IMPORTANT: 06_01_evaluate_run.sh requires CATEGORY as a MANDATORY 2nd positional arg (no auto-detect, unlike chaos/brats) -- each EVAL_ARGS entry below embeds it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

EVAL_SCRIPT="${HERE}/06_01_evaluate_run.sh"
EVAL_ARGS=(
    "open-ms_flair_baseline_20260706_061243 nnUNet"
    "open-ms_flair_synthseg_noEM_train100_val000_20260706_061243 auglab"
    "open-ms_flair_synthseg_EM_train100_val000_20260706_061243 auglab"
    "open-ms_flair_auglab_default_20260706_061243 auglab"
    "open-ms_flair_srcsm_20260709_072043 auglab"
    "open-ms_flair_auglabAug_v26_6_2_train050_val100_20260716_095413 auglab"
)
export CKPT_TAG=final
source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/run_all_evaluate_checkpoint_sweep.sh"
