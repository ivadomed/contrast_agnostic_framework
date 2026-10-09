#!/usr/bin/env bash
# Predict with AugLab default + V26_6_2 GPU transform @25% (train025_val000) on the
# CHAOS internal test set. Trainer is the default AugLab trainer (no aug at predict).
# Usage: bash 05_12_predict_auglabAug_v26_6_2_train025_val000.sh <RUN_ID> [FOLD] [MODALITY ...]
set -euo pipefail
METHOD="auglabAug_v26_6_2_train025_val000"
TRAINER="nnUNetTrainerCHAOSAugLabDefault"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
