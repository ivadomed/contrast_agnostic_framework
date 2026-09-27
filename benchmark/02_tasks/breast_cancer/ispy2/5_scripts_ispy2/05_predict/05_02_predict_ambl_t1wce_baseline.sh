#!/usr/bin/env bash
# Predict on I-SPY2 (122 usable bilateral cases, t1wce+t2w items) with the
# ambl t1wce-trained baseline model. I-SPY2 has no model of its own — see
# 05_01_predict_common.sh.
# Usage: bash 05_02_predict_ambl_t1wce_baseline.sh [RUN_ID] [FOLD]
set -euo pipefail
METHOD="baseline"
TRAINER="nnUNetTrainerAMBLBaseline"
CATEGORY="nnUNet"
export AMBL_TRAINING_CONTRAST="t1wce"
export AMBL_DATASET_ID="90"
RUN_ID="${1:-ambl_t1wce_baseline_20260903_003721}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
