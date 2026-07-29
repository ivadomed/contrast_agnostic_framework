#!/usr/bin/env bash
# Predict on BraTS-SSA 2024 with brats2024-glioma T1n-TRAINED auglabAug_v26_6_2 (train050_val000).
# Usage: bash 05_06_predict_brats_t1n_auglabAug_v26_6_2_train050_val000.sh [RUN_ID] [FOLD] [ITEM ...]
set -euo pipefail
METHOD="t1n_auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerBraTS2024GliomaAugLabDefault"
CATEGORY="auglab"
RUN_ID="${1:-brats2024-glioma_t1n_auglabAug_v26_6_2_train050_val000_20260710_040303}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
