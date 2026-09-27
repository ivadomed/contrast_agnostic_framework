#!/usr/bin/env bash
# Predict with T2w baseline_kmeans_label_remap_voronoi (causal ladder rung 4, noise fill)
# on the held-out BraTS test set.
#
# Usage:
#   bash 05_49_predict_t2w_baseline_kmeans_label_remap_voronoi.sh <RUN_ID> [FOLD] [CONTRAST ...]

set -euo pipefail
export TRAINING_CONTRAST="t2w"
METHOD="baseline_kmeans_label_remap_voronoi"
TRAINER="nnUNetTrainerBraTS2024GliomaT2wAugLabDefault"
DATASET_ID="052"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
