#!/usr/bin/env bash
# Predict with OUR METHOD (auglabAug_v26_6_2_train025_val100) on open-ms T1w across all
# test contrasts (FLAIR/T2W/T1W, folds 0 1 2).
# Usage: bash 05_16_predict_t1w_auglabAug_v26_6_2_train025_val100.sh <RUN_ID> [FOLD] [CONTRAST ...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t1w.sh"
METHOD="auglabAug_v26_6_2_train025_val100"
TRAINER="nnUNetTrainerOpenMSAugLabValSynth"
DATASET_ID="71"
CATEGORY="auglab"
export PREDICT_FOLDS="0 1 2"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
