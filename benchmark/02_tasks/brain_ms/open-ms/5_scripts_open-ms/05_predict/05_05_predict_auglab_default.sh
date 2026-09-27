#!/usr/bin/env bash
# Predict with auglab_default (augmentation, no synthesis) on the open-ms test set.
# Usage: bash 05_05_predict_auglab_default.sh <RUN_ID> [FOLD] [ITEM ...]
# Example: bash 05_05_predict_auglab_default.sh open-ms_auglab_default_<TS> all
set -euo pipefail
METHOD="auglab_default"
TRAINER="nnUNetTrainerOpenMSAugLabDefault"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
