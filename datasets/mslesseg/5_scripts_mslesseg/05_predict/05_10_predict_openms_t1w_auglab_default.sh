#!/usr/bin/env bash
# Predict on MSLesSeg with the open-ms T1w-TRAINED auglab_default model.
# Usage: bash 05_10_predict_openms_t1w_auglab_default.sh [RUN_ID] [FOLD] [ITEM ...]
set -euo pipefail
export OPENMS_TRAINING_CONTRAST="t1w"
export OPENMS_DATASET_ID="71"
export OPENMS_DS_NAME="Dataset071_OpenMS_T1W"
METHOD="t1w_auglab_default"
TRAINER="nnUNetTrainerOpenMSAugLabDefault"
CATEGORY="auglab"
RUN_ID="${1:-open-ms_t1w_auglab_default_20260708_083511}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
