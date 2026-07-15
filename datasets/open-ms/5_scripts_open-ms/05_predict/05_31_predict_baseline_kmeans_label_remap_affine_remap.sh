#!/usr/bin/env bash
# Predict with the K-means+label-remap+REAL-affine-remap rung (v26_6_2 minus
# Voronoi), trained by 04_33 with nnUNetTrainerOpenMSAugLabDualVal — trains
# directly as the _val000_ RUN_ID (its own checkpoint_best.pth already IS the
# clean/val000 result) and materializes ONE sibling _val100_ mirror at
# on_train_end, each a normal single-checkpoint run. Predict BOTH:
#     bash 05_31_predict_baseline_kmeans_label_remap_affine_remap.sh <RUN_ID with _val000_>
#     bash 05_31_predict_baseline_kmeans_label_remap_affine_remap.sh <RUN_ID with _val100_>
# (no CHECKPOINT=/PREDICT_OUTPUT_SUBDIR= override needed — each mirror's own
# checkpoint_best.pth is already the right one.)
# Usage: bash 05_31_predict_baseline_kmeans_label_remap_affine_remap.sh <RUN_ID> [FOLD] [ITEM ...]
set -euo pipefail
METHOD="baseline_kmeans_label_remap_affine_remap_train050_val000"
TRAINER="nnUNetTrainerOpenMSAugLabDualVal"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
