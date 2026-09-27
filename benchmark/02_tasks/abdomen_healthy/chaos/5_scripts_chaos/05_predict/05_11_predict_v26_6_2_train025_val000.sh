#!/usr/bin/env bash
# Predict with V26_6_2 train025_val000 on the CHAOS internal test set.
# Usage: bash 05_11_predict_v26_6_2_train025_val000.sh <RUN_ID> [FOLD] [MODALITY ...]
set -euo pipefail
METHOD="v26_6_2_train025_val000"
TRAINER="nnUNetTrainerCHAOSV26_6_2_train025_val000"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
