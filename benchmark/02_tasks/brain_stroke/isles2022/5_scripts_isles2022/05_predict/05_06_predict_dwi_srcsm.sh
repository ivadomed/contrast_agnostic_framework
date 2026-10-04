#!/usr/bin/env bash
# Predict with srcsm (dwi-trained, isles2022) on the held-out test set: dwi / adc / flair, all folds.
# Usage: bash 05_06_predict_dwi_srcsm.sh <RUN_ID> [FOLD] [ITEM ...]     (or let 05_24_run_all_predict_dwi.sh resolve the RUN_ID)
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="srcsm"
TRAINER="nnUNetTrainerISLES2022AugLabDefault"
CATEGORY="auglab"
DATASET_ID="${DATASET_ID_DWI}"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
