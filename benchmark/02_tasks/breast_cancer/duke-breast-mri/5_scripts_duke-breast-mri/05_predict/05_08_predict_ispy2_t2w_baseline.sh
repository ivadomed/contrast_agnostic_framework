#!/usr/bin/env bash
# Predict on duke-breast-mri (291 cases, t1wce only) with the ispy2 t2w-trained baseline model.
# See 05_01_predict_ispy2_common.sh.
# Usage: bash 05_08_predict_ispy2_t2w_baseline.sh [RUN_ID] [FOLD]
set -euo pipefail
METHOD="baseline"
TRAINER="nnUNetTrainerISPY2Baseline"
CATEGORY="nnUNet"
export ISPY2_TRAINING_CONTRAST="t2w"
export ISPY2_DATASET_ID="101"
RUN_ID="${1:-ispy2_t2w_baseline_20260905_163655}"
source "$(dirname "$0")/05_01_predict_ispy2_common.sh" "$RUN_ID" "${@:2}"
