#!/usr/bin/env bash
# Predict on MSD-SPLEEN CT with the CHAOS T2spir auglabAug v26_6_2 @50% (train050_val000,
# OURS) model. Uses chaos Dataset061_CHAOS_MR_T2spir checkpoints (pre-exports override
# env.sh defaults). train050_val000 uses the plain V26_6_2 trainer (no val-time synth).
# Usage: bash 05_17_predict_chaos_t2spir_auglabAug_v26_6_2_train050_val000.sh [CHAOS_RUN_ID] [FOLD] [MODALITY ...]
set -euo pipefail
export CHAOS_TRAINING_CONTRAST="t2spir"
export CHAOS_DATASET_ID="61"
export CHAOS_DS_NAME="Dataset061_CHAOS_MR_T2spir"
METHOD="t2spir_auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerCHAOSAugLabV26_6_2"
CATEGORY="auglab"
RUN_ID="${1:-chaos_t2spir_auglabAug_v26_6_2_train050_val000_20260710_054202}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
