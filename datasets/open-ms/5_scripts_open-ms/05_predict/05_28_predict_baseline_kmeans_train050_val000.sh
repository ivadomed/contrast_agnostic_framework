#!/usr/bin/env bash
# Predict with K-means intensity clustering ONLY, no label remap, @ train050_val000
# (nnUNet category, matches 04_30's placement) on the open-ms held-out test set.
# Baseline-anchored ladder, new rung between baseline and baseline_kmeans_label_remap.
# Usage: bash 05_28_predict_baseline_kmeans_train050_val000.sh <RUN_ID> [FOLD] [ITEM ...]
set -euo pipefail
METHOD="baseline_kmeans_train050_val000"
TRAINER="nnUNetTrainerOpenMSAugLabDefault"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
