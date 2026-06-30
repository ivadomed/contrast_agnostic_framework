#!/usr/bin/env bash
# Predict on TRUSTED (CT + US) with the CHAOS-TRAINED t2spir_auglabAug_v26_6_2 model (contrast: t2spir).
# TRUSTED has no model of its own — see 05_01_predict_common.sh. Default items: ct us.
# Usage: bash 05_14_predict_chaos_t2spir_auglabAug_v26_6_2.sh [CHAOS_RUN_ID] [FOLD] [ITEM ...]   (ITEM ∈ {ct,us})
set -euo pipefail
export CHAOS_TRAINING_CONTRAST="t2spir"
export CHAOS_DATASET_ID="61"
export CHAOS_DS_NAME="Dataset061_CHAOS_MR_T2spir"
METHOD="t2spir_auglabAug_v26_6_2"
TRAINER="nnUNetTrainerCHAOSAugLabValSynth"
CATEGORY="auglab"
RUN_ID="${1:-chaos_t2spir_auglabAug_v26_6_2_train025_val100_20260620_112633}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
