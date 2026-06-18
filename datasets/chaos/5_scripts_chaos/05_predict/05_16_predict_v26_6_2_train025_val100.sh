#!/usr/bin/env bash
# Predict with V26_6_2 (train_synth_prob=0.25, val_synth_prob=1.0) on the CHAOS test set.
# Usage: bash 05_16_predict_v26_6_2_train025_val100.sh <RUN_ID> [FOLD] [MODALITY ...]
set -euo pipefail
METHOD="v26_6_2_train025_val100"
TRAINER="nnUNetTrainerCHAOSV26_6_2_train025_val100"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
