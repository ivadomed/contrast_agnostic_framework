#!/usr/bin/env bash
# Predict on BraTS-SSA 2024 with the brats2024-glioma T2w-TRAINED baseline model.
# Usage: bash 05_08_predict_brats_t2w_baseline.sh [RUN_ID] [FOLD] [ITEM ...]
set -euo pipefail
export BRATS_TRAINING_CONTRAST="t2w"
export BRATS_DATASET_ID="052"
export BRATS_DS_NAME="Dataset052_BraTS2024GliomaT2w"
METHOD="t2w_baseline"
TRAINER="nnUNetTrainerBraTS2024GliomaT2wBaseline"
CATEGORY="nnUNet"
RUN_ID="${1:-brats2024-glioma_t2w_baseline_20260620_125115}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
