#!/usr/bin/env bash
# Predict on MSLesSeg with the open-ms AUGLAB_DEFAULT model (augmentation, no synthesis).
# Usage: bash 05_03_predict_openms_auglab_default.sh [RUN_ID] [FOLD] [ITEM ...]
set -euo pipefail
METHOD="auglab_default"
TRAINER="nnUNetTrainerOpenMSAugLabDefault"
CATEGORY="auglab"
RUN_ID="${1:-open-ms_flair_auglab_default_20260630_072742}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
