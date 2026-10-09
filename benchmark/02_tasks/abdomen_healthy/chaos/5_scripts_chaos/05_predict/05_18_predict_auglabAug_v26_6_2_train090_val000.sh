#!/usr/bin/env bash
# Predict with AugLab V26_6_2 GPU transform @90% (no val synthesis) on the CHAOS test set.
# Usage: bash 05_18_predict_auglabAug_v26_6_2_train090_val000.sh <RUN_ID> [FOLD] [MODALITY ...]
set -euo pipefail
METHOD="auglabAug_v26_6_2_train090_val000"
TRAINER="nnUNetTrainerCHAOSAugLabV26_6_2"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
