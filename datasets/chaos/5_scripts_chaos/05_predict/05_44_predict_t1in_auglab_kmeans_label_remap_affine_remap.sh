#!/usr/bin/env bash
# Predict with the auglab-anchored K-means+label-remap+REAL-affine-remap rung
# (auglabAug_v26_6_2 minus Voronoi), trained by 04_54 with
# nnUNetTrainerCHAOSAugLabDualVal — trains directly as the _val000_ RUN_ID
# (its own checkpoint_best.pth already IS the clean/val000 result) and
# materializes ONE sibling _val100_ mirror at on_train_end, each a normal
# single-checkpoint run. Predict BOTH:
#     bash 05_44_predict_t1in_auglab_kmeans_label_remap_affine_remap.sh <RUN_ID with _val000_>
#     bash 05_44_predict_t1in_auglab_kmeans_label_remap_affine_remap.sh <RUN_ID with _val100_>
# (no CHECKPOINT=/PREDICT_OUTPUT_SUBDIR= override needed — each mirror's own
# checkpoint_best.pth is already the right one.)
# Usage: bash 05_44_predict_t1in_auglab_kmeans_label_remap_affine_remap.sh <RUN_ID> [FOLD] [MODALITY ...]
set -euo pipefail
METHOD="auglab_kmeans_label_remap_affine_remap_train025_val000"
TRAINER="nnUNetTrainerCHAOSAugLabDualVal"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
