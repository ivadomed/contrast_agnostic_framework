#!/usr/bin/env bash
# Predict with AugLab default (val-aug variant) on the held-out BraTS test set.
# Results live under 8_results_brats2024-glioma/01_results/auglab_valaug/<RUN_ID>/.
#
# Usage:
#   bash 05_06_predict_auglab_default_valaug.sh <RUN_ID> [FOLD] [CONTRAST ...]
# Example:
#   bash 05_06_predict_auglab_default_valaug.sh auglab_default_valaug_20260610_000000 all

set -euo pipefail
METHOD="auglab_default_valaug"
TRAINER="nnUNetTrainerBraTS2024GliomaAugLabDefaultValAug"
DATASET_ID="051"
CATEGORY="auglab"   # common derives nnUNet_results from CATEGORY → 01_predictions/auglab/
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
