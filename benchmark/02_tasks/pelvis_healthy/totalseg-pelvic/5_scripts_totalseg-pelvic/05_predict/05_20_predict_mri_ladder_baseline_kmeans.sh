#!/usr/bin/env bash
# Predict with the MRI-trained ladder_baseline_kmeans model, over ct/mri.
# Usage: bash 05_20_predict_mri_ladder_baseline_kmeans.sh <RUN_ID> [FOLD] [items...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_mri.sh"
METHOD="baseline_kmeans"
TRAINER="nnUNetTrainerTotalsegPelvicAugLabDefault"
CATEGORY="auglab"
DATASET_ID="130"   # fixed — imagesTs_* consolidated under Dataset130, see 05_01 header
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
