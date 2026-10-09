#!/usr/bin/env bash
# Predict with V26_6_2_p50 (50% train synth) on the CHAOS internal test set.
# Usage: bash 05_07_predict_v26_6_2_50_100.sh <RUN_ID> [FOLD] [MODALITY ...]
set -euo pipefail
METHOD="v26_6_2_train050_val100"
TRAINER="nnUNetTrainerCHAOSV26_6_2_p50"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
