#!/usr/bin/env bash
# Predict on LLD-MMRI-HCC (T2w/DWI) with the atlas-liver-hcc-TRAINED baseline model
# (real-data-only control). LLD-MMRI-HCC has no model of its own — see 05_01_predict_common.sh.
# Usage: bash 05_02_predict_atlas_baseline.sh [RUN_ID] [FOLD] [ITEMS...]
set -euo pipefail
METHOD="baseline"
TRAINER="nnUNetTrainerAtlasLiverHCCBaseline"
CATEGORY="nnUNet"
RUN_ID="${1:-atlas-liver-hcc_t1w_baseline_20260804_062605}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
