#!/usr/bin/env bash
# Predict with baseline_kmeans_label_remap (dwi-trained, isles2022) on the held-out test set: dwi / flair, all folds.
# Usage: bash 05_10_predict_dwi_baseline_kmeans_label_remap.sh <RUN_ID> [FOLD] [ITEM ...]     (or let 05_24_run_all_predict_dwi.sh resolve the RUN_ID)
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="baseline_kmeans_label_remap"
TRAINER="nnUNetTrainerISLES2022AugLabDefault"
CATEGORY="auglab"
DATASET_ID="${DATASET_ID_DWI}"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
