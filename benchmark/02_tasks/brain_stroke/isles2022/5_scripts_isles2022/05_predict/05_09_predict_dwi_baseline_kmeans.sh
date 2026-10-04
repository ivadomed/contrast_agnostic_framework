#!/usr/bin/env bash
# Predict with baseline_kmeans (dwi-trained, isles2022) on the held-out test set: dwi / adc / flair, all folds.
# Usage: bash 05_09_predict_dwi_baseline_kmeans.sh <RUN_ID> [FOLD] [ITEM ...]     (or let 05_24_run_all_predict_dwi.sh resolve the RUN_ID)
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="baseline_kmeans"
TRAINER="nnUNetTrainerISLES2022AugLabDefault"
CATEGORY="auglab"
DATASET_ID="${DATASET_ID_DWI}"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
