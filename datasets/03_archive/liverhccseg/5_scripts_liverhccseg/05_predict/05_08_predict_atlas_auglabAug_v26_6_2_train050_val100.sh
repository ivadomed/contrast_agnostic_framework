#!/usr/bin/env bash
# Predict on LiverHccSeg with the atlas-liver-hcc DUAL-VAL run's synth-only/val100
# mirror checkpoint. Same trainer dir as val000 — see
# atlas-liver-hcc's atlas_liver_hcc/trainers/auglab_dualval.py.
# Usage: bash 05_08_predict_atlas_auglabAug_v26_6_2_train050_val100.sh [RUN_ID] [FOLD] [ITEMS...]
set -euo pipefail
METHOD="auglabAug_v26_6_2_train050_val100"
TRAINER="nnUNetTrainerAtlasLiverHCCAugLabDualVal"
CATEGORY="auglab"
RUN_ID="${1:-atlas-liver-hcc_t1w_auglabAug_v26_6_2_train050_val100_20260804_062605}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
