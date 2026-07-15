#!/usr/bin/env bash
# Predict with the K-means+label-remap rung (auglab-anchored), trained by
# 04_50 with nnUNetTrainerCHAOSAugLabDualVal — this ONE RUN_ID has TWO
# checkpoints, predict it TWICE:
#   val000 (clean-best, default):
#     bash 05_40_predict_t1in_auglab_kmeans_label_remap_dualval.sh <RUN_ID>
#   val100 (synth-best) — override checkpoint AND output subdir so it doesn't
#   collide with the val000 prediction:
#     CHECKPOINT=checkpoint_best_val100.pth PREDICT_OUTPUT_SUBDIR=val100 \
#       bash 05_40_predict_t1in_auglab_kmeans_label_remap_dualval.sh <RUN_ID>
# Usage: bash 05_40_predict_t1in_auglab_kmeans_label_remap_dualval.sh <RUN_ID> [FOLD] [MODALITY ...]
set -euo pipefail
METHOD="auglab_kmeans_label_remap_train025_dualval"
TRAINER="nnUNetTrainerCHAOSAugLabDualVal"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
