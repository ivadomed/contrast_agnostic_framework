#!/usr/bin/env bash
# GENERATED 2026-10-07 from 05_37_predict_t1in_baseline_kmeans_label_remap_voronoi.sh for the rung-4 lblvor run (scripts/cluster/rung4_lblvor): plain trainer
# nnUNetTrainerCHAOSAugLabDefault (the new run was trained without DualVal), otherwise identical.
# Predict with the K-means+label-remap+Voronoi rung, trained by 04_47 with
# nnUNetTrainerCHAOSAugLabDualVal — trains directly as the
# _val000_ RUN_ID (its own checkpoint_best.pth already IS the clean/val000
# result) and materializes ONE sibling _val100_ mirror at on_train_end, each
# a normal single-checkpoint run. Predict BOTH:
#     bash 05_37_predict_t1in_baseline_kmeans_label_remap_voronoi.sh <RUN_ID with _val000_>
#     bash 05_37_predict_t1in_baseline_kmeans_label_remap_voronoi.sh <RUN_ID with _val100_>
# (no CHECKPOINT=/PREDICT_OUTPUT_SUBDIR= override needed — each mirror's own
# checkpoint_best.pth is already the right one.)
# Usage: bash 05_37_predict_t1in_baseline_kmeans_label_remap_voronoi.sh <RUN_ID> [FOLD] [MODALITY ...]
set -euo pipefail
METHOD="baseline_kmeans_label_remap_voronoi_train050_val000_lblvor"
TRAINER="nnUNetTrainerCHAOSAugLabDefault"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
