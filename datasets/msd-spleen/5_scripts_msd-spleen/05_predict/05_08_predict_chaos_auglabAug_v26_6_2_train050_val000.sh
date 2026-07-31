#!/usr/bin/env bash
# Predict on MSD-SPLEEN CT with the CHAOS T1in auglabAug v26_6_2 @50% (train050_val000,
# OURS) model. train050_val000 uses the plain V26_6_2 trainer (no val-time synth).
# MSD-SPLEEN has no model of its own — see 05_01_predict_common.sh.
# Usage: bash 05_08_predict_chaos_auglabAug_v26_6_2_train050_val000.sh [CHAOS_RUN_ID] [FOLD] [MODALITY ...]
set -euo pipefail
METHOD="auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerCHAOSAugLabV26_6_2"
CATEGORY="auglab"
RUN_ID="${1:-chaos_t1in_auglabAug_v26_6_2_train050_val000_20260615_213615}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
