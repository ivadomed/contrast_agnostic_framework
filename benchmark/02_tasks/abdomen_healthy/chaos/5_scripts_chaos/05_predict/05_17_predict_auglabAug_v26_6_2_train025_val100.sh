#!/usr/bin/env bash
# Predict with AugLab V26_6_2 GPU transform @25% with val synthesis on the CHAOS test set.
# Usage: bash 05_17_predict_auglabAug_v26_6_2_train025_val100.sh <RUN_ID> [FOLD] [MODALITY ...]
set -euo pipefail
METHOD="auglabAug_v26_6_2_train025_val100"
TRAINER="nnUNetTrainerCHAOSAugLabValSynth"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
