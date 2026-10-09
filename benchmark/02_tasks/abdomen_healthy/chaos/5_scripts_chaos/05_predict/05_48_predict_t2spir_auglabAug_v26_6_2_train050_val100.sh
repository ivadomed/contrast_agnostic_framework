#!/usr/bin/env bash
# Predict with AugLab default + V26_6_2 GPU transform @50% train, 100% val synth
# (train050_val100) trained on CHAOS T2spir. Synth-only-val sibling of
# 05_34_predict_t2spir_auglabAug_v26_6_2_train050_val000.sh — trained with
# nnUNetTrainerCHAOSAugLabValSynth (see 04_58_train_t2spir_auglabAug_v26_6_2_train050_val100.sh).
# Usage: bash 05_48_predict_t2spir_auglabAug_v26_6_2_train050_val100.sh <RUN_ID> [FOLD] [MODALITY ...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t2spir.sh"
METHOD="auglabAug_v26_6_2_train050_val100"
TRAINER="nnUNetTrainerCHAOSAugLabValSynth"
DATASET_ID="61"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
