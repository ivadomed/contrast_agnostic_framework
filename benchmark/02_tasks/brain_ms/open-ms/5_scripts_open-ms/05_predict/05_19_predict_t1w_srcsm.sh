#!/usr/bin/env bash
# Predict with open-ms T1w srcsm (SRCSM SemRandConv-3D, added 7th comparison arm) across all contrasts.
# Usage: bash 05_19_predict_t1w_srcsm.sh <RUN_ID> [FOLD] [CONTRAST ...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t1w.sh"
METHOD="srcsm"
TRAINER="nnUNetTrainerOpenMSAugLabDefault"
DATASET_ID="71"
CATEGORY="auglab"
export PREDICT_FOLDS="0 1 2"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
