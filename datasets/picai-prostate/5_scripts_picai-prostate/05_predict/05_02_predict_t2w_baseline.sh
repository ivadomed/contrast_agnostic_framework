#!/usr/bin/env bash
# Predict with the baseline (no synthesis) (T2W-trained) on the picai-prostate
# held-out test set — T2W/ADC/HBV, folds 0 1 2.
# Usage: bash $(basename 05_predict/05_02_predict_t2w_baseline.sh) <RUN_ID> [FOLD] [ITEM ...]
set -euo pipefail
METHOD="baseline"
TRAINER="nnUNetTrainerPICAIProstateBaseline"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
