#!/usr/bin/env bash
# Shared predict template for totalseg-pelvic — sourced by 05_0X_predict_<method>[_mri].sh,
# NOT run directly. Thin shim: sources env.sh, sets the totalseg-pelvic config, and
# delegates to the shared driver datasets/00_commun_scripts/00_02_predict/predict_common.sh.
#
# Predicts with totalseg-pelvic's OWN trained models (CT- or MRI-trained, whichever
# wrapper set TRAINER/DATASET_ID) over TWO items, both consolidated under Dataset130's
# raw tree by 05_00_build_test_inputs.py:
#   ct   — held-out CT test cases   (own-contrast for a CT-trained model, cross-contrast for MRI-trained)
#   mri  — held-out MRI test cases  (mirror of the above)
#
# PREDICT_DATASET_ID_DEFAULT is fixed at 130 REGARDLESS of which per-modality Dataset
# actually trained the model — see 05_00_build_test_inputs.py's docstring.
set -euo pipefail
cd "${PROJECT_ROOT:?source 00_utils/env.sh (or env_mri.sh) BEFORE this shim}"

PREDICT_MODE="own"
PREDICT_JOB_PREFIX="totalseg_pelvic_predict"
PREDICT_LOG_PREFIX="predict"
PREDICT_ITEMS_DEFAULT="ct mri"
PREDICT_FOLD_DEFAULT="all"
PREDICT_DATASET_ID_DEFAULT="130"
PREDICT_TIME="01:00:00"
PREDICT_EXTRA_FLAGS="-npp 4 -nps 2"

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
