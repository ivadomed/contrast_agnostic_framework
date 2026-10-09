#!/usr/bin/env bash
# Predict with V26_6_2 train050_val000 on the CHAOS internal test set.
# Usage: bash 05_10_predict_v26_6_2_train050_val000.sh <RUN_ID> [FOLD] [MODALITY ...]
set -euo pipefail
METHOD="v26_6_2_train050_val000"
TRAINER="nnUNetTrainerCHAOSV26_6_2_train050_val000"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
