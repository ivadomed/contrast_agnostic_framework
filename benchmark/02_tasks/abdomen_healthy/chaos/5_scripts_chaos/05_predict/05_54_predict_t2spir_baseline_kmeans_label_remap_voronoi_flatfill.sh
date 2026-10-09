#!/usr/bin/env bash
# GENERATED 2026-10-08 from 05_52_predict_t2spir_baseline_kmeans_label_remap_voronoi_lblvor.sh for the rung-4.5 flat-fill run (scripts/cluster/rung45_flat): plain trainer
# nnUNetTrainerCHAOSAugLabDefault (the new run was trained without DualVal), otherwise identical.
# Predict with the K-means+label-remap+Voronoi rung (T2spir), trained by
# 04_57 with nnUNetTrainerCHAOSAugLabDualVal — trains directly as the
# _val000_ RUN_ID (its own checkpoint_best.pth already IS the clean/val000
# result) and materializes ONE sibling _val100_ mirror at on_train_end, each
# a normal single-checkpoint run. Predict BOTH:
#     bash 05_47_predict_t2spir_baseline_kmeans_label_remap_voronoi.sh <RUN_ID with _val000_>
#     bash 05_47_predict_t2spir_baseline_kmeans_label_remap_voronoi.sh <RUN_ID with _val100_>
# (no CHECKPOINT=/PREDICT_OUTPUT_SUBDIR= override needed — each mirror's own
# checkpoint_best.pth is already the right one.)
# Usage: bash 05_47_predict_t2spir_baseline_kmeans_label_remap_voronoi.sh <RUN_ID> [FOLD] [MODALITY ...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t2spir.sh"
METHOD="baseline_kmeans_label_remap_voronoi_train050_val000_flatfill"
TRAINER="nnUNetTrainerCHAOSAugLabDefault"
DATASET_ID="61"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
