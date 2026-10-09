#!/usr/bin/env bash
# Predict with T1n AugLabAug + V26_6_2 (25% train synth / 100% val synth) on the held-out BraTS test set.
#
# Usage:
#   bash 05_15_predict_t1n_auglabAug_v26_6_2_train025_val100.sh <RUN_ID> [FOLD] [CONTRAST ...]
# Example:
#   bash 05_15_predict_t1n_auglabAug_v26_6_2_train025_val100.sh brats2024-glioma_t1n_auglabAug_v26_6_2_train025_val100_20260622_044535

set -euo pipefail
METHOD="auglabAug_v26_6_2_train025_val100"
TRAINER="nnUNetTrainerBraTS2024GliomaAugLabValSynth"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
