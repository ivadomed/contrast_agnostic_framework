#!/usr/bin/env bash
# Predict on BraTS-SSA 2024 with OUR METHOD, T2w-TRAINED (brats2024-glioma auglabAug_v26_6_2,
# train050_val100) -- the dual-val run's val100 mirror (TRAINER=...T2wAugLabDualVal).
# Usage: bash 05_13_predict_brats_t2w_auglabAug_v26_6_2_train050_val100.sh [RUN_ID] [FOLD] [ITEM ...]
set -euo pipefail
export BRATS_TRAINING_CONTRAST="t2w"
export BRATS_DATASET_ID="052"
export BRATS_DS_NAME="Dataset052_BraTS2024GliomaT2w"
METHOD="t2w_auglabAug_v26_6_2_train050_val100"
TRAINER="nnUNetTrainerBraTS2024GliomaT2wAugLabDualVal"
CATEGORY="auglab"
RUN_ID="${1:-brats2024-glioma_t2w_auglabAug_v26_6_2_train050_val100_20260725_113540}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
