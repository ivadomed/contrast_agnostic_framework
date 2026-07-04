#!/usr/bin/env bash
# Predict with synthseg_noEM on the open-ms held-out test set (FLAIR/T2W/T1W, all folds).
# Usage: bash 05_06_predict_synthseg_noEM.sh <RUN_ID> [FOLD] [ITEM ...]
set -euo pipefail
METHOD="synthseg_noEM_train100_val000"
TRAINER="nnUNetTrainerOpenMSAugLabDefault"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
