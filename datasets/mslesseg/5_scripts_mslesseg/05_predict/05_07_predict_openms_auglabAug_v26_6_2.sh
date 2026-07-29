#!/usr/bin/env bash
# Predict on MSLesSeg with OUR METHOD (open-ms auglabAug_v26_6_2, train025_val100).
# Usage: bash 05_07_predict_openms_auglabAug_v26_6_2.sh [RUN_ID] [FOLD] [ITEM ...]
set -euo pipefail
METHOD="auglabAug_v26_6_2_train025_val100"
TRAINER="nnUNetTrainerOpenMSAugLabValSynth"
CATEGORY="auglab"
RUN_ID="${1:-open-ms_flair_auglabAug_v26_6_2_train025_val100_20260706_061243}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
