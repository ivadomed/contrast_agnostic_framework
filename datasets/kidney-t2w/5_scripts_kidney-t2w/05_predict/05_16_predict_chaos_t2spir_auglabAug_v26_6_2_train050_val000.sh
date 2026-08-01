#!/usr/bin/env bash
# Predict on KIDNEY-T2W with the CHAOS-TRAINED T2spir auglabAug v26_6_2 @50% (train050_val000, OURS) model.
# KIDNEY-T2W has no model of its own — see 05_01_predict_common.sh.
# Usage: bash 05_16_predict_chaos_t2spir_auglabAug_v26_6_2_train050_val000.sh [CHAOS_RUN_ID] [FOLD] [MODALITY ...]
set -euo pipefail
export CHAOS_TRAINING_CONTRAST="t2spir"
export CHAOS_DATASET_ID="61"
export CHAOS_DS_NAME="Dataset061_CHAOS_MR_T2spir"
METHOD="t2spir_auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerCHAOSAugLabV26_6_2"
CATEGORY="auglab"
RUN_ID="${1:-chaos_t2spir_auglabAug_v26_6_2_train050_val000_20260710_054202}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
