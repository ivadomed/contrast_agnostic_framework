#!/usr/bin/env bash
# Predict with AugLab default + V26_6_2 GPU transform @50% (train050_val000) on the
# CHAOS internal test set. Trainer applies no augmentation at predict time.
# Usage: bash 05_09_predict_auglabAug_v26_6_2_train050_val000.sh <RUN_ID> [FOLD] [MODALITY ...]
set -euo pipefail
METHOD="auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerCHAOSAugLabV26_6_2"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
