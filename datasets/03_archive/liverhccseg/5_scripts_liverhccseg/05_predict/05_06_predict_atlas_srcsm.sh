#!/usr/bin/env bash
# Predict on LiverHccSeg with the atlas-liver-hcc-TRAINED srcsm model.
# Usage: bash 05_06_predict_atlas_srcsm.sh [RUN_ID] [FOLD] [ITEMS...]
set -euo pipefail
METHOD="srcsm"
TRAINER="nnUNetTrainerAtlasLiverHCCAugLabDefault"
CATEGORY="auglab"
RUN_ID="${1:-atlas-liver-hcc_t1w_srcsm_20260804_062605}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
