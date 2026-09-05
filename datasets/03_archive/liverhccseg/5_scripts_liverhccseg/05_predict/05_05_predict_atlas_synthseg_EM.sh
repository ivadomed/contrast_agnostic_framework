#!/usr/bin/env bash
# Predict on LiverHccSeg with the atlas-liver-hcc-TRAINED synthseg_EM model.
# Usage: bash 05_05_predict_atlas_synthseg_EM.sh [RUN_ID] [FOLD] [ITEMS...]
set -euo pipefail
METHOD="synthseg_EM"
TRAINER="nnUNetTrainerAtlasLiverHCCAugLabDefault"
CATEGORY="auglab"
RUN_ID="${1:-atlas-liver-hcc_t1w_synthseg_EM_20260804_062605}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
