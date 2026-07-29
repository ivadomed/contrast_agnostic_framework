#!/usr/bin/env bash
# Predict on MSLesSeg with OUR METHOD, T1w-TRAINED (open-ms auglabAug_v26_6_2, train025_val100).
# Usage: bash 05_14_predict_openms_t1w_auglabAug_v26_6_2.sh [RUN_ID] [FOLD] [ITEM ...]
set -euo pipefail
export OPENMS_TRAINING_CONTRAST="t1w"
export OPENMS_DATASET_ID="71"
export OPENMS_DS_NAME="Dataset071_OpenMS_T1W"
METHOD="t1w_auglabAug_v26_6_2_train025_val100"
TRAINER="nnUNetTrainerOpenMSAugLabValSynth"
CATEGORY="auglab"
RUN_ID="${1:-open-ms_t1w_auglabAug_v26_6_2_train025_val100_20260708_083711}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
