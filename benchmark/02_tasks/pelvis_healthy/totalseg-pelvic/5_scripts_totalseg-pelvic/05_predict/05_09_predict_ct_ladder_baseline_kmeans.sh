#!/usr/bin/env bash
# Predict with the CT-trained ladder_baseline_kmeans model, over ct/mri.
# Usage: bash 05_09_predict_ct_ladder_baseline_kmeans.sh <RUN_ID> [FOLD] [items...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="baseline_kmeans"
TRAINER="nnUNetTrainerTotalsegPelvicAugLabDefault"
CATEGORY="auglab"
DATASET_ID="130"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
