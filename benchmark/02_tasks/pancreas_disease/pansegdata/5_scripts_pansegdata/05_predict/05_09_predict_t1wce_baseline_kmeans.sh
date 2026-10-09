#!/usr/bin/env bash
# Predict with baseline_kmeans (t1wce-trained, pansegdata) on the held-out test set: t1wce / t2w, all folds.
# Usage: bash 05_09_predict_t1wce_baseline_kmeans.sh <RUN_ID> [FOLD] [ITEM ...]     (or let 05_24_run_all_predict_t1wce.sh resolve the RUN_ID)
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="baseline_kmeans"
TRAINER="nnUNetTrainerPANSEGDATAAugLabDefault"
CATEGORY="auglab"
DATASET_ID="${DATASET_ID_T1WCE}"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
