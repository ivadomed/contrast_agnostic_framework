#!/usr/bin/env bash
# Predict on BraTS-SSA 2024 with the brats2024-glioma T1n-TRAINED auglab_default model.
# Usage: bash 05_03_predict_brats_t1n_auglab_default.sh [RUN_ID] [FOLD] [ITEM ...]
set -euo pipefail
METHOD="t1n_auglab_default"
TRAINER="nnUNetTrainerBraTS2024GliomaAugLabDefault"
CATEGORY="auglab"
RUN_ID="${1:-brats2024-glioma_t1n_auglab_default_20260622_044535}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
