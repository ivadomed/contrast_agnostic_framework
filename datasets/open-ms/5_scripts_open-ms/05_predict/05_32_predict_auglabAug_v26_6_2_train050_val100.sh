#!/usr/bin/env bash
# Predict with OUR METHOD (auglabAug_v26_6_2_train050_val100) on the open-ms held-out
# test set (FLAIR/T2W/T1W, all folds). Synth-only-val sibling of
# 05_20_predict_auglabAug_v26_6_2_train050_val000.sh — same method/checkpoint family,
# trained with nnUNetTrainerOpenMSAugLabValSynth (see 04_33_train_auglabAug_v26_6_2_train050_val100.sh).
# Usage: bash 05_32_predict_auglabAug_v26_6_2_train050_val100.sh <RUN_ID> [FOLD] [ITEM ...]
# Example: bash 05_32_predict_auglabAug_v26_6_2_train050_val100.sh open-ms_flair_auglabAug_v26_6_2_train050_val100_<TS> all
set -euo pipefail
METHOD="auglabAug_v26_6_2_train050_val100"
TRAINER="nnUNetTrainerOpenMSAugLabValSynth"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
