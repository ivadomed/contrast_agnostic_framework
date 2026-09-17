#!/usr/bin/env bash
# Predict on the FOV-matched PDDCA CT test set with the toothfairy2 CBCT-trained
# ladder rung4 (noise fill) model (cross-MODALITY: CBCT -> CT). See 05_01_predict_common.sh.
# Usage: bash 05_11_predict_baseline_kmeans_label_remap_voronoi.sh <RUN_ID> [FOLD]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="baseline_kmeans_label_remap_voronoi"
TRAINER="nnUNetTrainerToothFairy2AugLabDefault"
CATEGORY="auglab"
DATASET_ID="${TF2_DATASET_ID}"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
