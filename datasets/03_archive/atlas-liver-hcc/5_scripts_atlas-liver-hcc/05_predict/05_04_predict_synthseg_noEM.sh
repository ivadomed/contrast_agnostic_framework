#!/usr/bin/env bash
# Predict with the atlas-liver-hcc synthseg_noEM model on the held-out T1w test set.
# Usage: bash 05_04_predict_synthseg_noEM.sh <RUN_ID> [FOLD]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="synthseg_noEM"
TRAINER="nnUNetTrainerAtlasLiverHCCAugLabDefault"
DATASET_ID="80"
CATEGORY="auglab"
export PREDICT_FOLDS="0 1 2"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
