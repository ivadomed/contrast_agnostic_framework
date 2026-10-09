#!/usr/bin/env bash
# Predict with K-means intensity clustering ONLY, no label remap (nnUNet
# category, matches 04_29's placement) on the open-ms held-out test set
# (FLAIR/T2W/T1W, folds 0-2 — see TRAIN_FOLDS in 04_29). Baseline-anchored
# ladder, new rung between baseline and baseline_kmeans_label_remap.
# Usage: bash 05_27_predict_baseline_kmeans_train050_val100.sh <RUN_ID> [FOLD] [ITEM ...]
set -euo pipefail
METHOD="baseline_kmeans_train050_val100"
TRAINER="nnUNetTrainerOpenMSAugLabValSynth"
CATEGORY="nnUNet"
export PREDICT_FOLDS="0 1 2"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
