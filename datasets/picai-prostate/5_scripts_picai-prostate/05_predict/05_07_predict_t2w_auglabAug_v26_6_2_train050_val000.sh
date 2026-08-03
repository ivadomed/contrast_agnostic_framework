#!/usr/bin/env bash
# Predict with OUR METHOD, clean/val000 checkpoint (T2W-trained) on the picai-prostate
# held-out test set — T2W/ADC/HBV, folds 0 1 2.
#
# TRAINER IS THE DUALVAL CLASS ON PURPOSE: both the _val000 run and its hard-linked
# _val100 mirror live under nnUNetTrainerPICAIProstateAugLabDualVal's directory name
# (see 04_06's header) — using AugLabDefault/AugLabValSynth here finds no checkpoint.
# Usage: bash $(basename 05_predict/05_07_predict_t2w_auglabAug_v26_6_2_train050_val000.sh) <RUN_ID> [FOLD] [ITEM ...]
set -euo pipefail
METHOD="t2w_auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerPICAIProstateAugLabDualVal"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
