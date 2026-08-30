#!/usr/bin/env bash
# Predict with K-means intensity clustering ONLY, no label remap, @ train050_val000
# (nnUNet category, matches 04_35's placement) on the open-ms T1w held-out test set.
# Baseline-anchored ladder rung 2, T1w sibling of 05_28 (the FLAIR version).
# Usage: bash 05_35_predict_t1w_baseline_kmeans_train050_val000.sh <RUN_ID> [FOLD] [CONTRAST ...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t1w.sh"
METHOD="baseline_kmeans_train050_val000"
TRAINER="nnUNetTrainerOpenMSAugLabDefault"
DATASET_ID="71"
CATEGORY="nnUNet"
export PREDICT_FOLDS="0 1 2"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
