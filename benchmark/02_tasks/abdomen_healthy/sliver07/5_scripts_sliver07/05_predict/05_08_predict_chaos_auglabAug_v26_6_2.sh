#!/usr/bin/env bash
# Predict on SLIVER07 CT with the CHAOS-TRAINED auglabAug v26_6_2 model.
# SLIVER07 has no model of its own — see 05_01_predict_common.sh.
# Usage: bash 05_08_predict_chaos_auglabAug_v26_6_2.sh [CHAOS_RUN_ID] [FOLD] [MODALITY ...]
set -euo pipefail
METHOD="auglabAug_v26_6_2"
TRAINER="nnUNetTrainerCHAOSAugLabValSynth"
CATEGORY="auglab"
RUN_ID="${1:-chaos_t1in_auglabAug_v26_6_2_train025_val100_20260616_200514}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
