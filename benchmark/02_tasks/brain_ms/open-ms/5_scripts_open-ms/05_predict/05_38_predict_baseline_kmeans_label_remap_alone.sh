#!/usr/bin/env bash
# Predict with K-means+noise-fill ALONE (no Voronoi, no AugLab intensity suite —
# nnUNet category, matches 04_10/05_07's placement) on the open-ms held-out test set
# (FLAIR/T2W/T1W, folds 0-2 — see TRAIN_FOLDS in 04_10). Baseline-anchored ladder,
# rung 2 of 4 (SynthSeg-EM analog).
# Usage: bash 05_11_predict_baseline_kmeans_label_remap_alone.sh <RUN_ID> [FOLD] [ITEM ...]
set -euo pipefail
METHOD="baseline_kmeans_label_remap_train050_val100"
TRAINER="nnUNetTrainerOpenMSAugLabValSynth"
CATEGORY="nnUNet"
# Trained on folds 0-2 only (TRAIN_FOLDS in 04_10) — no fold-3 checkpoint exists.
export PREDICT_FOLDS="0 1 2"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
