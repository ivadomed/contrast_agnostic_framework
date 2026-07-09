#!/usr/bin/env bash
# Predict with open-ms T1w synthseg_EM across all test contrasts (FLAIR/T2W/T1W).
# Usage: bash 05_14_predict_t1w_synthseg_EM.sh <RUN_ID> [FOLD] [CONTRAST ...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t1w.sh"
METHOD="synthseg_EM_train100_val000"
TRAINER="nnUNetTrainerOpenMSAugLabDefault"
DATASET_ID="71"
CATEGORY="auglab"
export PREDICT_FOLDS="0 1 2"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
