#!/usr/bin/env bash
# Predict with synthseg_EM on the open-ms held-out test set (FLAIR/T2W/T1W, all folds).
# Usage: bash 05_02_predict_synthseg_EM.sh <RUN_ID> [FOLD] [ITEM ...]
# Example: bash 05_02_predict_synthseg_EM.sh open-ms_synthseg_EM_train100_val000_<TS> all
set -euo pipefail
METHOD="synthseg_EM_train100_val000"
TRAINER="nnUNetTrainerOpenMSAugLabDefault"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
