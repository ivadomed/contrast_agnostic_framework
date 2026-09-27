#!/usr/bin/env bash
# Predict with the train050_val000 variant of the Voronoi-ablation arm
# (auglab_kmeans_label_remap_train050_val000, auglab category, matches 04_27's
# placement) on the open-ms held-out test set (FLAIR/T2W/T1W, folds 0-2 — see
# TRAIN_FOLDS in 04_27). Auglab-anchored ladder, rung 2 of 4.
# Usage: bash 05_25_predict_auglab_kmeans_label_remap_train050_val000.sh <RUN_ID> [FOLD] [ITEM ...]
set -euo pipefail
METHOD="auglab_kmeans_label_remap_train050_val000"
TRAINER="nnUNetTrainerOpenMSAugLabDefault"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
