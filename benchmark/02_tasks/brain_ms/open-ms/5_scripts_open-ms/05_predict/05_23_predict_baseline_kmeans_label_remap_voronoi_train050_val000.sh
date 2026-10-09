#!/usr/bin/env bash
# Predict with the train050_val000 variant of K-means+Voronoi+noise-fill ALONE
# (nnUNet category, matches 04_25's placement) on the open-ms held-out test set
# (FLAIR/T2W/T1W, folds 0-2 — see TRAIN_FOLDS in 04_25). Baseline-anchored
# ladder, rung 3 of 4.
# Usage: bash 05_23_predict_baseline_kmeans_label_remap_voronoi_train050_val000.sh <RUN_ID> [FOLD] [ITEM ...]
set -euo pipefail
METHOD="baseline_kmeans_label_remap_voronoi_train050_val000"
TRAINER="nnUNetTrainerOpenMSAugLabDefault"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
