#!/usr/bin/env bash
# Predict on BraTS-SSA 2024 with OUR METHOD, T1n-TRAINED (brats2024-glioma auglabAug_v26_6_2,
# train050_val100). Regular single-checkpoint AugLabValSynth run (NOT dual-val) -- see
# brats2024-glioma's own 05_24_predict_t1n_auglabAug_v26_6_2_train050_val100.sh.
# Usage: bash 05_07_predict_brats_t1n_auglabAug_v26_6_2_train050_val100.sh [RUN_ID] [FOLD] [ITEM ...]
set -euo pipefail
METHOD="t1n_auglabAug_v26_6_2_train050_val100"
TRAINER="nnUNetTrainerBraTS2024GliomaAugLabValSynth"
CATEGORY="auglab"
RUN_ID="${1:-brats2024-glioma_t1n_auglabAug_v26_6_2_train050_val100_20260725_113540}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
