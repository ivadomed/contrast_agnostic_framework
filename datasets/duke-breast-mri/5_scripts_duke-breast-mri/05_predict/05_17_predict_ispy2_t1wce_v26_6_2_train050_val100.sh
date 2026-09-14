#!/usr/bin/env bash
# Predict on duke-breast-mri with the ispy2 t1wce causal-ablation ladder model (v26_6_2_train050_val100).
# Usage: bash 05_17_predict_ispy2_t1wce_v26_6_2_train050_val100.sh [RUN_ID] [FOLD]
set -euo pipefail
METHOD="v26_6_2_train050_val100"
TRAINER="nnUNetTrainerISPY2AugLabValSynth"
CATEGORY="nnUNet"
export ISPY2_TRAINING_CONTRAST="t1wce"
export ISPY2_DATASET_ID="100"
RUN_ID="${1:-ispy2_t1wce_v26_6_2_train050_val100_20260905_163655}"
source "$(dirname "$0")/05_01_predict_ispy2_common.sh" "$RUN_ID" "${@:2}"
