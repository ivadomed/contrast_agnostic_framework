#!/usr/bin/env bash
# Predict with v26_6_2 ALONE @ train050_val000 (nnUNet category, matches
# 04_26's placement) on the open-ms held-out test set. Baseline-anchored
# ladder, rung 4 of 4.
# Usage: bash 05_24_predict_v26_6_2_train050_val000.sh <RUN_ID> [FOLD] [ITEM ...]
set -euo pipefail
METHOD="v26_6_2_train050_val000"
TRAINER="nnUNetTrainerOpenMSAugLabDefault"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
