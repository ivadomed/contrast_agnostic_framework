#!/usr/bin/env bash
# Re-predict the t1in suite's EXISTING checkpoint_best RUN_IDs at checkpoint_final.pth,
# for the checkpoint_best vs checkpoint_final comparison (see
# 06_evaluate/configs/chaos_t1in_03_results_*_ckpt.yaml). Only srcsm and auglabAug_v26_6_2 have checkpoint_final.pth on disk -- baseline/synthseg_noEM/synthseg_EM/auglab_default never kept one; left out here, documented blanks in the comparison configs.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

METHOD_SCRIPTS=(
    "${HERE}/05_32_predict_srcsm.sh"
    "${HERE}/05_14_predict_auglabAug_v26_6_2_train050_val100.sh"
)
METHOD_RUN_IDS=(
    "chaos_t1in_srcsm_20260710_011817"
    "chaos_t1in_auglabAug_v26_6_2_train050_val100_20260616_112420"
)
export CHECKPOINT=checkpoint_final.pth
source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/run_all_predict_checkpoint_sweep.sh"
