#!/usr/bin/env bash
# Predict with V26_6_2 (whole-image + Voronoi sub-parc + label-wise synth) on the
# held-out BraTS test set, single fold, across all (or selected) contrasts.
#
# Usage:
#   bash 05_03_predict_v26_6_2.sh <RUN_ID> [FOLD] [CONTRAST ...]
# Examples:
#   bash 05_03_predict_v26_6_2.sh brats2024-glioma_t1n_v26_6_2_train090_val000_20260608_003445              # fold 0, all contrasts
#   bash 05_03_predict_v26_6_2.sh brats2024-glioma_t1n_v26_6_2_train090_val000_20260608_003445 1            # fold 1, all contrasts
#   bash 05_03_predict_v26_6_2.sh brats2024-glioma_t1n_v26_6_2_train090_val000_20260608_003445 0 t1n t2f    # fold 0, two contrasts

set -euo pipefail
METHOD="v26_6_2"
TRAINER="nnUNetTrainerBraTS2024GliomaV26_6_2"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_predict_common.sh" "$@"
