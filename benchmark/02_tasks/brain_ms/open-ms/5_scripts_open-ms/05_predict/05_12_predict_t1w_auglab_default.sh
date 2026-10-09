#!/usr/bin/env bash
# Predict with open-ms T1w auglab_default (augmentation, no synthesis) across all contrasts.
# Usage: bash 05_12_predict_t1w_auglab_default.sh <RUN_ID> [FOLD] [CONTRAST ...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t1w.sh"
METHOD="auglab_default"
TRAINER="nnUNetTrainerOpenMSAugLabDefault"
DATASET_ID="71"
CATEGORY="auglab"
export PREDICT_FOLDS="0 1 2"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
