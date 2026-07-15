#!/usr/bin/env bash
# Predict with auglabAug_v26_6_2 minus Voronoi @ train025_val100 (auglab
# category, matches 04_36's placement) on the open-ms held-out test set.
# Auglab-anchored ladder, extra arm.
# Usage: bash 05_34_predict_auglab_kmeans_label_remap_affine_remap_train025_val100.sh <RUN_ID> [FOLD] [ITEM ...]
set -euo pipefail
METHOD="auglab_kmeans_label_remap_affine_remap_train025_val100"
TRAINER="nnUNetTrainerOpenMSAugLabValSynth"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
