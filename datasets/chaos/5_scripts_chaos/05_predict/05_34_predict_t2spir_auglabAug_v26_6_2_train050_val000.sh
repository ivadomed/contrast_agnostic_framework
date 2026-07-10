#!/usr/bin/env bash
# Predict with AugLab default + V26_6_2 GPU transform @50% (train050_val000) trained on
# CHAOS T2spir. Trainer applies no augmentation at predict time.
# Usage: bash 05_34_predict_t2spir_auglabAug_v26_6_2_train050_val000.sh <RUN_ID> [FOLD] [MODALITY ...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t2spir.sh"
METHOD="auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerCHAOSAugLabV26_6_2"
DATASET_ID="61"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
