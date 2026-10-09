#!/usr/bin/env bash
# Predict with v26_6_2 alone (nnUNet category) on the open-ms held-out test set.
# Usage: bash 05_07_predict_v26_6_2.sh <RUN_ID> [FOLD] [ITEM ...]
set -euo pipefail
METHOD="v26_6_2_train050_val100"
TRAINER="nnUNetTrainerOpenMSV26_6_2"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
