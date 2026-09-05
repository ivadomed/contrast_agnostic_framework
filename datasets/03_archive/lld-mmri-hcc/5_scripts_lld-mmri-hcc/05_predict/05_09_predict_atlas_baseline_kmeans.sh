#!/usr/bin/env bash
# Predict on the atlas-liver-hcc causal-ablation ladder rung 2 (K-means intensity
# clustering only). See 05_02_predict_atlas_baseline.sh header for the general
# cross-dataset predict note.
# Usage: bash 05_09_predict_atlas_baseline_kmeans.sh [RUN_ID] [FOLD] [ITEMS...]
set -euo pipefail
METHOD="baseline_kmeans"
TRAINER="nnUNetTrainerAtlasLiverHCCAugLabDefault"
CATEGORY="auglab"
RUN_ID="${1:-atlas-liver-hcc_t1w_baseline_kmeans_20260829_165445}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
