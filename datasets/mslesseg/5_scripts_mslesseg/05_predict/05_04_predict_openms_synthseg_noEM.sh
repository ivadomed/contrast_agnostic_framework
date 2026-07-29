#!/usr/bin/env bash
# Predict on MSLesSeg with the open-ms SYNTHSEG_noEM model.
# Usage: bash 05_04_predict_openms_synthseg_noEM.sh [RUN_ID] [FOLD] [ITEM ...]
set -euo pipefail
METHOD="synthseg_noEM_train100_val000"
TRAINER="nnUNetTrainerOpenMSAugLabDefault"
CATEGORY="auglab"
RUN_ID="${1:-open-ms_flair_synthseg_noEM_train100_val000_20260706_061243}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
