#!/usr/bin/env bash
# Predict with the T1c-trained ladder rung 3 (K-means + label remap) on the held-out BraTS test set (70 cases),
# all folds, across all 4 contrasts (t1n t1c t2w t1c — cross-contrast is the headline result).
#
# Usage:
#   bash 05_79_predict_t1c_baseline_kmeans_label_remap.sh <RUN_ID> [FOLD] [CONTRAST ...]
# Example:
#   bash 05_79_predict_t1c_baseline_kmeans_label_remap.sh brats2024-glioma_t1c_baseline_kmeans_label_remap_<TS> all

set -euo pipefail
export TRAINING_CONTRAST="t1c"
METHOD="t1c_baseline_kmeans_label_remap"
TRAINER="nnUNetTrainerBraTS2024GliomaT1cAugLabDefault"
DATASET_ID="054"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
