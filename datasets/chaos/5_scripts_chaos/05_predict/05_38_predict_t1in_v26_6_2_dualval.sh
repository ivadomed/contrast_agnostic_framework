#!/usr/bin/env bash
# Predict with v26_6_2 alone, trained by 04_48 with
# nnUNetTrainerCHAOSAugLabDualVal — this ONE RUN_ID has TWO checkpoints,
# predict it TWICE:
#   val000 (clean-best, default):
#     bash 05_38_predict_t1in_v26_6_2_dualval.sh <RUN_ID>
#   val100 (synth-best) — override checkpoint AND output subdir so it doesn't
#   collide with the val000 prediction:
#     CHECKPOINT=checkpoint_best_val100.pth PREDICT_OUTPUT_SUBDIR=val100 \
#       bash 05_38_predict_t1in_v26_6_2_dualval.sh <RUN_ID>
# Usage: bash 05_38_predict_t1in_v26_6_2_dualval.sh <RUN_ID> [FOLD] [MODALITY ...]
set -euo pipefail
METHOD="v26_6_2_train050_dualval"
TRAINER="nnUNetTrainerCHAOSAugLabDualVal"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
