#!/usr/bin/env bash
# Predict with the T2f/FLAIR-trained OURS DualVal (val100 mirror = synth-best checkpoint) on the held-out BraTS test set (70 cases),
# all folds, across all 4 contrasts (t1n t1c t2w t2f — cross-contrast is the headline result).
#
# Usage:
#   bash 05_56_predict_t2f_auglabAug_v26_6_2_train050_val100_dualval.sh <RUN_ID> [FOLD] [CONTRAST ...]
# Example:
#   bash 05_56_predict_t2f_auglabAug_v26_6_2_train050_val100_dualval.sh brats2024-glioma_t2f_auglabAug_v26_6_2_train050_val100_<TS> all

set -euo pipefail
export TRAINING_CONTRAST="t2f"
METHOD="t2f_auglabAug_v26_6_2_train050_val100"
TRAINER="nnUNetTrainerBraTS2024GliomaT2fAugLabDualVal"
DATASET_ID="053"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
