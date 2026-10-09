#!/usr/bin/env bash
# Predict with synthseg_noEM (t2w-trained, pansegdata) on the held-out test set: t1wce / t2w, all folds.
# Usage: bash 05_15_predict_t2w_synthseg_noEM.sh <RUN_ID> [FOLD] [ITEM ...]     (or let 05_25_run_all_predict_t2w.sh resolve the RUN_ID)
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t2w.sh"
METHOD="synthseg_noEM"
TRAINER="nnUNetTrainerPANSEGDATAAugLabDefault"
CATEGORY="auglab"
DATASET_ID="${DATASET_ID_T2W}"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
