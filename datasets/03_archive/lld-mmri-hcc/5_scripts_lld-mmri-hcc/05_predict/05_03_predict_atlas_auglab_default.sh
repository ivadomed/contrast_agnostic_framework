#!/usr/bin/env bash
# Predict on LLD-MMRI-HCC (T2w/DWI) with the atlas-liver-hcc-TRAINED auglab_default model.
# Usage: bash 05_03_predict_atlas_auglab_default.sh [RUN_ID] [FOLD] [ITEMS...]
set -euo pipefail
METHOD="auglab_default"
TRAINER="nnUNetTrainerAtlasLiverHCCAugLabDefault"
CATEGORY="auglab"
RUN_ID="${1:-atlas-liver-hcc_t1w_auglab_default_20260804_062605}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
