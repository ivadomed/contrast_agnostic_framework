#!/usr/bin/env bash
# Predict on MS3SEG with the open-ms T1w-TRAINED synthseg_noEM model.
# Usage: bash 05_11_predict_openms_t1w_synthseg_noEM_train100_val000.sh [RUN_ID] [FOLD] [ITEM ...]
set -euo pipefail
export OPENMS_TRAINING_CONTRAST="t1w"
export OPENMS_DATASET_ID="71"
export OPENMS_DS_NAME="Dataset071_OpenMS_T1W"
METHOD="t1w_synthseg_noEM_train100_val000"
TRAINER="nnUNetTrainerOpenMSAugLabDefault"
CATEGORY="auglab"
RUN_ID="${1:-open-ms_t1w_synthseg_noEM_train100_val000_20260708_083541}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
