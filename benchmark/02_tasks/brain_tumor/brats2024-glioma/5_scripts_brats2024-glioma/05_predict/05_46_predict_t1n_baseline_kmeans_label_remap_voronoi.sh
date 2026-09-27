#!/usr/bin/env bash
# Predict with T1n baseline_kmeans_label_remap_voronoi (causal ladder rung 4, noise fill)
# on the held-out BraTS test set.
#
# Usage:
#   bash 05_46_predict_t1n_baseline_kmeans_label_remap_voronoi.sh <RUN_ID> [FOLD] [CONTRAST ...]

set -euo pipefail
METHOD="baseline_kmeans_label_remap_voronoi"
TRAINER="nnUNetTrainerBraTS2024GliomaAugLabDefault"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
