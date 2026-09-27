#!/usr/bin/env bash
# Predict with the baseline (no synthesis) on the open-ms held-out test set.
# Usage: bash 05_04_predict_baseline.sh <RUN_ID> [FOLD] [ITEM ...]
# Example: bash 05_04_predict_baseline.sh open-ms_baseline_<TS> all
set -euo pipefail
METHOD="baseline"
TRAINER="nnUNetTrainerOpenMSBaseline"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
