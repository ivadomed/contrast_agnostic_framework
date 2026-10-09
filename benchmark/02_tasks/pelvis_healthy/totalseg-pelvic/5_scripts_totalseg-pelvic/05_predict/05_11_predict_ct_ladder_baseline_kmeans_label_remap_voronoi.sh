#!/usr/bin/env bash
# Predict with the CT-trained ladder_baseline_kmeans_label_remap_voronoi model, over ct/mri.
# Usage: bash 05_11_predict_ct_ladder_baseline_kmeans_label_remap_voronoi.sh <RUN_ID> [FOLD] [items...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="baseline_kmeans_label_remap_voronoi"
TRAINER="nnUNetTrainerTotalsegPelvicAugLabDefault"
CATEGORY="auglab"
DATASET_ID="130"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
