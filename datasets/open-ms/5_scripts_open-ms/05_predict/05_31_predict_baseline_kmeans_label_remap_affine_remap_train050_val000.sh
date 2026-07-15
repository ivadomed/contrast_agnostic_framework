#!/usr/bin/env bash
# Predict with v26_6_2 minus Voronoi @ train050_val000 (nnUNet category,
# matches 04_33's placement) on the open-ms held-out test set.
# Baseline-anchored ladder, extra arm.
# Usage: bash 05_31_predict_baseline_kmeans_label_remap_affine_remap_train050_val000.sh <RUN_ID> [FOLD] [ITEM ...]
set -euo pipefail
METHOD="baseline_kmeans_label_remap_affine_remap_train050_val000"
TRAINER="nnUNetTrainerOpenMSAugLabDefault"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
