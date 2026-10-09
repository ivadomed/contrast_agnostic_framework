#!/usr/bin/env bash
# Predict the T2w DUAL-VAL run's synth-only/val100 checkpoint — i.e. the "_val100_"
# MIRROR RUN_ID auto-materialized by 04_32's dual-val run at on_train_end. The mirror
# lives under the SAME nnUNetTrainerBraTS2024GliomaT2wAugLabDualVal trainer dir as its
# val000 sibling, so predict MUST use TRAINER=...T2wAugLabDualVal (NOT ...AugLabValSynth).
# See brats2024_glioma/trainers/auglab_dualval.py.
#
# Usage:
#   bash 05_23_predict_t2w_auglabAug_v26_6_2_train050_val100_dualval.sh <RUN_ID with _val100_> [FOLD] [CONTRAST ...]
# Example:
#   bash 05_23_predict_t2w_auglabAug_v26_6_2_train050_val100_dualval.sh brats2024-glioma_t2w_auglabAug_v26_6_2_train050_val100_<TS> all

set -euo pipefail
export TRAINING_CONTRAST="t2w"
METHOD="t2w_auglabAug_v26_6_2_train050_val100"
TRAINER="nnUNetTrainerBraTS2024GliomaT2wAugLabDualVal"
DATASET_ID="052"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
