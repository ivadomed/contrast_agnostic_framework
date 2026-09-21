#!/usr/bin/env bash
# Predict with the T2f/FLAIR-trained ladder rung 3 (K-means + label remap) on the held-out BraTS test set (70 cases),
# all folds, across all 4 contrasts (t1n t1c t2w t2f — cross-contrast is the headline result).
#
# Usage:
#   bash 05_59_predict_t2f_baseline_kmeans_label_remap.sh <RUN_ID> [FOLD] [CONTRAST ...]
# Example:
#   bash 05_59_predict_t2f_baseline_kmeans_label_remap.sh brats2024-glioma_t2f_baseline_kmeans_label_remap_<TS> all

set -euo pipefail
export TRAINING_CONTRAST="t2f"
METHOD="t2f_baseline_kmeans_label_remap"
TRAINER="nnUNetTrainerBraTS2024GliomaT2fAugLabDefault"
DATASET_ID="053"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
