#!/usr/bin/env bash
# Predict with AugLab default on the held-out BraTS test set.
# Results live under 8_results_brats2024-glioma/01_results/auglab/<RUN_ID>/.
#
# Usage:
#   bash 05_05_predict_auglab_default.sh <RUN_ID> [FOLD] [CONTRAST ...]
# Examples:
#   bash 05_05_predict_auglab_default.sh brats2024-glioma_auglab_default_20260609_220932
#   bash 05_05_predict_auglab_default.sh brats2024-glioma_auglab_default_20260609_220932 all
#   bash 05_05_predict_auglab_default.sh brats2024-glioma_auglab_default_20260609_220932 2

set -euo pipefail
METHOD="auglab_default"
TRAINER="nnUNetTrainerBraTS2024GliomaAugLabDefault"
DATASET_ID="051"
CATEGORY="auglab"   # common derives nnUNet_results from CATEGORY → 01_predictions/auglab/
source "$(dirname "$0")/05_predict_common.sh" "$@"
