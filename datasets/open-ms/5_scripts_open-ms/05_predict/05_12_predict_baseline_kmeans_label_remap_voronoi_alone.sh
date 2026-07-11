#!/usr/bin/env bash
# Predict with K-means+Voronoi+noise-fill ALONE (no AugLab intensity suite — nnUNet
# category, matches 04_11/05_07's placement) on the open-ms held-out test set
# (FLAIR/T2W/T1W, folds 0-2 — see TRAIN_FOLDS in 04_11). Baseline-anchored ladder,
# rung 3 of 4.
# Usage: bash 05_12_predict_baseline_kmeans_label_remap_voronoi_alone.sh <RUN_ID> [FOLD] [ITEM ...]
set -euo pipefail
METHOD="baseline_kmeans_label_remap_voronoi_train050_val100"
TRAINER="nnUNetTrainerOpenMSAugLabValSynth"
CATEGORY="nnUNet"
# Trained on folds 0-2 only (TRAIN_FOLDS in 04_11) — no fold-3 checkpoint exists.
export PREDICT_FOLDS="0 1 2"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
