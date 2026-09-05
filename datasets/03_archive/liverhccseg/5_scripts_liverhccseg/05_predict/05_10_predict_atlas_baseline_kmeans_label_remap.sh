#!/usr/bin/env bash
# Predict on the atlas-liver-hcc causal-ablation ladder rung 3 (+ label remap).
# See 05_02_predict_atlas_baseline.sh header for the general cross-dataset predict note.
# Usage: bash 05_10_predict_atlas_baseline_kmeans_label_remap.sh [RUN_ID] [FOLD] [ITEMS...]
set -euo pipefail
METHOD="baseline_kmeans_label_remap"
TRAINER="nnUNetTrainerAtlasLiverHCCAugLabDefault"
CATEGORY="auglab"
RUN_ID="${1:-atlas-liver-hcc_t1w_baseline_kmeans_label_remap_20260829_165445}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
