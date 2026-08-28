#!/usr/bin/env bash
# Predict on LLD-MMRI-HCC (T2w/DWI) with the atlas-liver-hcc DUAL-VAL run's clean/val000
# checkpoint (OURS). MUST use TRAINER=...AugLabDualVal (NOT ...AugLabDefault) — see
# atlas-liver-hcc's atlas_liver_hcc/trainers/auglab_dualval.py.
# Usage: bash 05_07_predict_atlas_auglabAug_v26_6_2_train050_val000.sh [RUN_ID] [FOLD] [ITEMS...]
set -euo pipefail
METHOD="auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerAtlasLiverHCCAugLabDualVal"
CATEGORY="auglab"
RUN_ID="${1:-atlas-liver-hcc_t1w_auglabAug_v26_6_2_train050_val000_20260804_062605}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
