#!/usr/bin/env bash
# Predict with T1n baseline_kmeans (causal ladder rung 2) on the held-out BraTS test set.
#
# Usage:
#   bash 05_44_predict_t1n_baseline_kmeans.sh <RUN_ID> [FOLD] [CONTRAST ...]

set -euo pipefail
METHOD="baseline_kmeans"
TRAINER="nnUNetTrainerBraTS2024GliomaAugLabDefault"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
