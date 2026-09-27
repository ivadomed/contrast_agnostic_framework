#!/usr/bin/env bash
# Predict with T1n baseline_kmeans_label_remap (causal ladder rung 3) on the held-out BraTS test set.
#
# Usage:
#   bash 05_45_predict_t1n_baseline_kmeans_label_remap.sh <RUN_ID> [FOLD] [CONTRAST ...]

set -euo pipefail
METHOD="baseline_kmeans_label_remap"
TRAINER="nnUNetTrainerBraTS2024GliomaAugLabDefault"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
