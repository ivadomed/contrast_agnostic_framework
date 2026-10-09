#!/usr/bin/env bash
# Predict on duke-breast-mri with the ispy2 t1wce causal-ablation ladder model (baseline_kmeans_label_remap_voronoi).
# Usage: bash 05_16_predict_ispy2_t1wce_baseline_kmeans_label_remap_voronoi.sh [RUN_ID] [FOLD]
set -euo pipefail
METHOD="baseline_kmeans_label_remap_voronoi"
TRAINER="nnUNetTrainerISPY2AugLabDefault"
CATEGORY="auglab"
export ISPY2_TRAINING_CONTRAST="t1wce"
export ISPY2_DATASET_ID="100"
RUN_ID="${1:-ispy2_t1wce_baseline_kmeans_label_remap_voronoi_20260905_163655}"
source "$(dirname "$0")/05_01_predict_ispy2_common.sh" "$RUN_ID" "${@:2}"
