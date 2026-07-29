#!/usr/bin/env bash
# Predict on MS3SEG with the open-ms FLAIR-TRAINED auglab_default model.
# Usage: bash 05_03_predict_openms_auglab_default.sh [RUN_ID] [FOLD] [ITEM ...]
set -euo pipefail
METHOD="auglab_default"
TRAINER="nnUNetTrainerOpenMSAugLabDefault"
CATEGORY="auglab"
RUN_ID="${1:-open-ms_flair_auglab_default_20260706_061243}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
