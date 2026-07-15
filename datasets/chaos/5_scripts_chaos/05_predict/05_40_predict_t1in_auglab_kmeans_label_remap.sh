#!/usr/bin/env bash
# Predict with the K-means+label-remap rung (auglab-anchored), trained by
# 04_50 with nnUNetTrainerCHAOSAugLabDualVal — training materializes TWO separate
# mirror RUN_IDs at on_train_end (same <TS>, "_dualval_" swapped for
# "_val000_"/"_val100_"), each a normal single-checkpoint run. Predict BOTH:
#     bash 05_40_predict_t1in_auglab_kmeans_label_remap_dualval.sh <RUN_ID with _val000_>
#     bash 05_40_predict_t1in_auglab_kmeans_label_remap_dualval.sh <RUN_ID with _val100_>
# (no CHECKPOINT=/PREDICT_OUTPUT_SUBDIR= override needed — each mirror's own
# checkpoint_best.pth is already the right one.)
# Usage: bash 05_40_predict_t1in_auglab_kmeans_label_remap_dualval.sh <RUN_ID> [FOLD] [MODALITY ...]
set -euo pipefail
METHOD="auglab_kmeans_label_remap_train025_dualval"
TRAINER="nnUNetTrainerCHAOSAugLabDualVal"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
