#!/usr/bin/env bash
# Predict with srcsm (T2W-trained) on the picai-prostate
# held-out test set — T2W/ADC/HBV, folds 0 1 2.
# Usage: bash $(basename 05_predict/05_06_predict_t2w_srcsm.sh) <RUN_ID> [FOLD] [ITEM ...]
set -euo pipefail
METHOD="srcsm"
TRAINER="nnUNetTrainerPICAIProstateAugLabDefault"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
