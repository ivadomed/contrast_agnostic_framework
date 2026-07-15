#!/usr/bin/env bash
# Predict with auglabAug_v26_6_2 (OURS), trained by 04_52 with
# nnUNetTrainerCHAOSAugLabDualVal — training materializes TWO separate
# mirror RUN_IDs at on_train_end (same <TS>, "_dualval_" swapped for
# "_val000_"/"_val100_"), each a normal single-checkpoint run. Predict BOTH:
#     bash 05_42_predict_t1in_auglabAug_v26_6_2_dualval.sh <RUN_ID with _val000_>
#     bash 05_42_predict_t1in_auglabAug_v26_6_2_dualval.sh <RUN_ID with _val100_>
# (no CHECKPOINT=/PREDICT_OUTPUT_SUBDIR= override needed — each mirror's own
# checkpoint_best.pth is already the right one.)
# Usage: bash 05_42_predict_t1in_auglabAug_v26_6_2_dualval.sh <RUN_ID> [FOLD] [MODALITY ...]
set -euo pipefail
METHOD="auglabAug_v26_6_2_train025_dualval"
TRAINER="nnUNetTrainerCHAOSAugLabDualVal"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
