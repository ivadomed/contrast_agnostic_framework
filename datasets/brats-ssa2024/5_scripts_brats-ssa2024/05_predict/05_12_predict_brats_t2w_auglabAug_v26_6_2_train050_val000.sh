#!/usr/bin/env bash
# Predict on BraTS-SSA 2024 with brats2024-glioma T2w-TRAINED auglabAug_v26_6_2 (train050_val000).
# This is the DUAL-VAL run's val000 mirror -- both val000/val100 live under the SAME
# nnUNetTrainerBraTS2024GliomaT2wAugLabDualVal trainer dir (verified: same RUN_ID
# timestamp as the val100 mirror below) -- see brats2024-glioma's own
# 05_23_predict_t2w_auglabAug_v26_6_2_train050_val100_dualval.sh / [[project_dualval_trainer]].
# Usage: bash 05_12_predict_brats_t2w_auglabAug_v26_6_2_train050_val000.sh [RUN_ID] [FOLD] [ITEM ...]
set -euo pipefail
export BRATS_TRAINING_CONTRAST="t2w"
export BRATS_DATASET_ID="052"
export BRATS_DS_NAME="Dataset052_BraTS2024GliomaT2w"
METHOD="t2w_auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerBraTS2024GliomaT2wAugLabDualVal"
CATEGORY="auglab"
RUN_ID="${1:-brats2024-glioma_t2w_auglabAug_v26_6_2_train050_val000_20260725_113540}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
