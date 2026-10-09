#!/usr/bin/env bash
# Predict with AugLabAug V26_6_2 (train025/val100) trained on CHAOS T2spir.
# Usage: bash 05_26_predict_t2spir_auglabAug_v26_6_2_train025_val100.sh <RUN_ID> [FOLD] [MODALITY ...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t2spir.sh"
METHOD="auglabAug_v26_6_2_train025_val100"
TRAINER="nnUNetTrainerCHAOSAugLabValSynth"
DATASET_ID="61"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
