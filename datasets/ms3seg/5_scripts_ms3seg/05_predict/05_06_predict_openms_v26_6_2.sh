#!/usr/bin/env bash
# Predict on MS3SEG with the open-ms FLAIR-TRAINED v26_6_2-ALONE model.
# Usage: bash 05_06_predict_openms_v26_6_2_train050_val100.sh [RUN_ID] [FOLD] [ITEM ...]
set -euo pipefail
METHOD="v26_6_2_train050_val100"
TRAINER="nnUNetTrainerOpenMSV26_6_2"
CATEGORY="nnUNet"
RUN_ID="${1:-open-ms_flair_v26_6_2_train050_val100_20260706_061243}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
