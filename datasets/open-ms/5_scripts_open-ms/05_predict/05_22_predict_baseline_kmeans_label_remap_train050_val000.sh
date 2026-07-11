#!/usr/bin/env bash
# Predict with the train050_val000 variant of K-means+noise-fill ALONE (no
# Voronoi, nnUNet category, matches 04_24's placement) on the open-ms held-out
# test set (FLAIR/T2W/T1W, folds 0-2 — see TRAIN_FOLDS in 04_24). Baseline-
# anchored ladder, rung 2 of 4.
# Usage: bash 05_22_predict_baseline_kmeans_label_remap_train050_val000.sh <RUN_ID> [FOLD] [ITEM ...]
set -euo pipefail
METHOD="baseline_kmeans_label_remap_train050_val000"
TRAINER="nnUNetTrainerOpenMSAugLabDefault"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
