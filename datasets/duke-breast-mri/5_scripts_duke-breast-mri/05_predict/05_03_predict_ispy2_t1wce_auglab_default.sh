#!/usr/bin/env bash
# Predict on duke-breast-mri (291 cases, t1wce only) with the ispy2 t1wce-trained auglab_default model.
# See 05_01_predict_ispy2_common.sh.
# Usage: bash 05_03_predict_ispy2_t1wce_auglab_default.sh [RUN_ID] [FOLD]
set -euo pipefail
METHOD="auglab_default"
TRAINER="nnUNetTrainerISPY2AugLabDefault"
CATEGORY="auglab"
export ISPY2_TRAINING_CONTRAST="t1wce"
export ISPY2_DATASET_ID="100"
RUN_ID="${1:-ispy2_t1wce_auglab_default_20260905_163655}"
source "$(dirname "$0")/05_01_predict_ispy2_common.sh" "$RUN_ID" "${@:2}"
