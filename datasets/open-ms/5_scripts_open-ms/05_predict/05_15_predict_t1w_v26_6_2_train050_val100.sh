#!/usr/bin/env bash
# Predict with open-ms T1w v26_6_2 alone (nnUNet category) across all test contrasts.
# Usage: bash 05_15_predict_t1w_v26_6_2_train050_val100.sh <RUN_ID> [FOLD] [CONTRAST ...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t1w.sh"
METHOD="v26_6_2_train050_val100"
TRAINER="nnUNetTrainerOpenMSV26_6_2"
DATASET_ID="71"
CATEGORY="nnUNet"
export PREDICT_FOLDS="0 1 2"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
