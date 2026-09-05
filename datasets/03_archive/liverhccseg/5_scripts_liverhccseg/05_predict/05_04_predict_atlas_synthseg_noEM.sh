#!/usr/bin/env bash
# Predict on LiverHccSeg with the atlas-liver-hcc-TRAINED synthseg_noEM model.
# Usage: bash 05_04_predict_atlas_synthseg_noEM.sh [RUN_ID] [FOLD] [ITEMS...]
set -euo pipefail
METHOD="synthseg_noEM"
TRAINER="nnUNetTrainerAtlasLiverHCCAugLabDefault"
CATEGORY="auglab"
RUN_ID="${1:-atlas-liver-hcc_t1w_synthseg_noEM_20260804_062605}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
