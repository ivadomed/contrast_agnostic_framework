#!/usr/bin/env bash
# Predict with OUR METHOD (auglabAug_v26_6_2_train025_val100) on the open-ms held-out
# test set (FLAIR/T2W/T1W, all folds).
# Usage: bash 05_03_predict_auglabAug_v26_6_2.sh <RUN_ID> [FOLD] [ITEM ...]
# Example: bash 05_03_predict_auglabAug_v26_6_2.sh open-ms_auglabAug_v26_6_2_train025_val100_<TS> all
set -euo pipefail
METHOD="auglabAug_v26_6_2_train025_val100"
TRAINER="nnUNetTrainerOpenMSAugLabValSynth"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
