#!/usr/bin/env bash
# Predict the T2w DUAL-VAL run's clean/val000 checkpoint (AugLab + V26_6_2 @50% train).
# The dual-val run (04_32) produces BOTH val000 and val100 checkpoints in ONE training,
# so BOTH mirror RUN_IDs live under the nnUNetTrainerBraTS2024GliomaT2wAugLabDualVal
# trainer dir. Predict MUST therefore use TRAINER=...T2wAugLabDualVal — NOT
# ...T2wAugLabDefault (the separate-training val000 wrapper 05_21) nor ...T2wAugLabValSynth.
# See brats2024_glioma/trainers/auglab_dualval.py.
#
# Usage:
#   bash 05_22_predict_t2w_auglabAug_v26_6_2_train050_val000_dualval.sh <RUN_ID with _val000_> [FOLD] [CONTRAST ...]
# Example:
#   bash 05_22_predict_t2w_auglabAug_v26_6_2_train050_val000_dualval.sh brats2024-glioma_t2w_auglabAug_v26_6_2_train050_val000_<TS> all

set -euo pipefail
export TRAINING_CONTRAST="t2w"
METHOD="t2w_auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerBraTS2024GliomaT2wAugLabDualVal"
DATASET_ID="052"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
