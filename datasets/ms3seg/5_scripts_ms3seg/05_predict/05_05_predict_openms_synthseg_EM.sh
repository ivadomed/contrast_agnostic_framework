#!/usr/bin/env bash
# Predict on MS3SEG with the open-ms FLAIR-TRAINED synthseg_EM model.
# Usage: bash 05_05_predict_openms_synthseg_EM_train100_val000.sh [RUN_ID] [FOLD] [ITEM ...]
set -euo pipefail
METHOD="synthseg_EM_train100_val000"
TRAINER="nnUNetTrainerOpenMSAugLabDefault"
CATEGORY="auglab"
RUN_ID="${1:-open-ms_flair_synthseg_EM_train100_val000_20260706_061243}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
