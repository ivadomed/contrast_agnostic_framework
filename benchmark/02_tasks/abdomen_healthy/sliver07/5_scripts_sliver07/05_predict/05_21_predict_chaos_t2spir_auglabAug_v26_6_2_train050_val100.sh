#!/usr/bin/env bash
# Predict on SLIVER07 CT with the CHAOS T2spir auglabAug v26_6_2 @50% (train050_val100)
# model — the val000/val100 significance comparison's cross-dataset arm. Uses chaos
# Dataset061_CHAOS_MR_T2spir checkpoints (pre-exports override env.sh defaults).
# train050_val100 uses the ValSynth trainer (val-time synth), unlike train050_val000's
# plain V26_6_2 trainer.
# Usage: bash 05_21_predict_chaos_t2spir_auglabAug_v26_6_2_train050_val100.sh [CHAOS_RUN_ID] [FOLD] [MODALITY ...]
set -euo pipefail
export CHAOS_TRAINING_CONTRAST="t2spir"
export CHAOS_DATASET_ID="61"
export CHAOS_DS_NAME="Dataset061_CHAOS_MR_T2spir"
METHOD="t2spir_auglabAug_v26_6_2_train050_val100"
TRAINER="nnUNetTrainerCHAOSAugLabValSynth"
CATEGORY="auglab"
RUN_ID="${1:-chaos_t2spir_auglabAug_v26_6_2_train050_val100_20260723_194053}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
