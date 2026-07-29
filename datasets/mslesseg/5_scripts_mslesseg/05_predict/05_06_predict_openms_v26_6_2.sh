#!/usr/bin/env bash
# Predict on MSLesSeg with the open-ms v26_6_2-ALONE model (train050_val100).
# Usage: bash 05_06_predict_openms_v26_6_2.sh [RUN_ID] [FOLD] [ITEM ...]
set -euo pipefail
METHOD="v26_6_2_train050_val100"
TRAINER="nnUNetTrainerOpenMSV26_6_2"
CATEGORY="nnUNet"
RUN_ID="${1:-open-ms_flair_v26_6_2_train050_val100_20260630_102558}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
