#!/usr/bin/env bash
# Predict on KIDNEY-T2W with the CHAOS-TRAINED auglabAug v26_6_2 @50% (train050_val000, OURS) model.
# KIDNEY-T2W has no model of its own — see 05_01_predict_common.sh.
# Usage: bash 05_07_predict_chaos_auglabAug_v26_6_2_train050_val000.sh [CHAOS_RUN_ID] [FOLD] [MODALITY ...]
set -euo pipefail
METHOD="auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerCHAOSAugLabV26_6_2"
CATEGORY="auglab"
RUN_ID="${1:-chaos_t1in_auglabAug_v26_6_2_train050_val000_20260615_213615}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
