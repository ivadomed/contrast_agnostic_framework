#!/usr/bin/env bash
# Predict with K-means intensity clustering ONLY, no label remap (auglab
# category, matches 04_31's placement) on the open-ms held-out test set
# (FLAIR/T2W/T1W, folds 0-2 — see TRAIN_FOLDS in 04_31). Auglab-anchored
# ladder, new rung between auglab_default and auglab_kmeans_label_remap.
# Usage: bash 05_29_predict_auglab_kmeans_train025_val100.sh <RUN_ID> [FOLD] [ITEM ...]
set -euo pipefail
METHOD="auglab_kmeans_train025_val100"
TRAINER="nnUNetTrainerOpenMSAugLabValSynth"
CATEGORY="auglab"
export PREDICT_FOLDS="0 1 2"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
