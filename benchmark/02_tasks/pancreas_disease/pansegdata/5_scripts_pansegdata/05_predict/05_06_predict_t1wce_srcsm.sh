#!/usr/bin/env bash
# Predict with srcsm (t1wce-trained, pansegdata) on the held-out test set: t1wce / t2w, all folds.
# Usage: bash 05_06_predict_t1wce_srcsm.sh <RUN_ID> [FOLD] [ITEM ...]     (or let 05_24_run_all_predict_t1wce.sh resolve the RUN_ID)
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="srcsm"
TRAINER="nnUNetTrainerPANSEGDATAAugLabDefault"
CATEGORY="auglab"
DATASET_ID="${DATASET_ID_T1WCE}"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
