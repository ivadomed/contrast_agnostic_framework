#!/usr/bin/env bash
# Predict on MS3SEG with the open-ms T1w-TRAINED v26_6_2-ALONE model.
# Usage: bash 05_13_predict_openms_t1w_v26_6_2_train050_val100.sh [RUN_ID] [FOLD] [ITEM ...]
set -euo pipefail
export OPENMS_TRAINING_CONTRAST="t1w"
export OPENMS_DATASET_ID="71"
export OPENMS_DS_NAME="Dataset071_OpenMS_T1W"
METHOD="t1w_v26_6_2_train050_val100"
TRAINER="nnUNetTrainerOpenMSV26_6_2"
CATEGORY="nnUNet"
RUN_ID="${1:-open-ms_t1w_v26_6_2_train050_val100_20260708_083641}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
