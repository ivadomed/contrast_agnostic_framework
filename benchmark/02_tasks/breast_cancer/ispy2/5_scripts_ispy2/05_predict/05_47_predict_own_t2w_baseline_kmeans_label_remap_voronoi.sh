#!/usr/bin/env bash
# Predict on ispy2's own test cases with the T2W ladder rung-4 model
# (+Voronoi sub-parcellation, noise fill). Was missing 2026-09-06.
# Usage: bash 05_47_predict_own_t2w_baseline_kmeans_label_remap_voronoi.sh <RUN_ID> [FOLD]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t2w.sh"
METHOD="baseline_kmeans_label_remap_voronoi"
TRAINER="nnUNetTrainerISPY2AugLabDefault"
CATEGORY="auglab"
DATASET_ID="${DATASET_ID_T2W}"
source "$(dirname "$0")/05_20_predict_common.sh" "$@"
