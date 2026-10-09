#!/usr/bin/env bash
# Predict with V26_6_2 on the CHAOS internal test set across modalities.
# Usage: bash 05_03_predict_v26_6_2.sh <RUN_ID> [FOLD] [MODALITY ...]
# Example: bash 05_03_predict_v26_6_2.sh v26_6_2_20260614_000000 all
set -euo pipefail
METHOD="v26_6_2"
TRAINER="nnUNetTrainerCHAOSV26_6_2"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
