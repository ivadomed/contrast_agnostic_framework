#!/usr/bin/env bash
# Predict with K-means intensity clustering ONLY, no label remap, @ train050_val000
# (auglab category, matches 04_32's placement) on the open-ms held-out test set.
# Auglab-anchored ladder, new rung between auglab_default and auglab_kmeans_label_remap.
# Usage: bash 05_30_predict_auglab_kmeans_train050_val000.sh <RUN_ID> [FOLD] [ITEM ...]
set -euo pipefail
METHOD="auglab_kmeans_train050_val000"
TRAINER="nnUNetTrainerOpenMSAugLabDefault"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
