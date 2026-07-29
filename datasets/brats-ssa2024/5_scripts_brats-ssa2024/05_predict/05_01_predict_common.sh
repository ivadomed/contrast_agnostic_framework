#!/usr/bin/env bash
# ============================================================================
#  BraTS-SSA 2024 prediction -- USES MODELS TRAINED ON ANOTHER DATASET (brats2024-glioma).
#  BraTS-SSA 2024 is EVALUATION-ONLY (see README.md): no native models. These load
#  brats2024-glioma-trained checkpoints (T1n- or T2w-trained, per SOURCE_PREFIX/
#  BRATS_TRAINING_CONTRAST) and run them over all 95 glioma cases to measure glioma-
#  segmentation domain-randomization generalization to an independent SSA cohort.
#
#  Default item list is "t1n t1c t2w t2f" -- brats2024-glioma trains single-channel
#  models per contrast and tests cross-contrast on all 4, same as its own held-out test.
# ============================================================================
# Thin shim: sources env.sh, sets the brats-ssa2024 cross-dataset config, and delegates
# to the shared driver datasets/00_commun_scripts/00_02_predict/predict_common.sh.
# Sourced by 05_0X_predict_brats_<method>.sh (which sets METHOD/TRAINER/CATEGORY/RUN_ID).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

SOURCE_PREFIX="BRATS"
PREDICT_MODE="cross"
PREDICT_JOB_PREFIX="brats_ssa_predict"
PREDICT_LOG_PREFIX="brats_ssa_predict"
PREDICT_ITEMS_DEFAULT="t1n t1c t2w t2f"
PREDICT_FOLD_DEFAULT="all"

# BraTS-SSA volumes are standard BraTS size (240x240x155) -- same as brats2024-glioma's
# own test set, so reuse its own predict timing (own predict_common.sh uses 0:45:00).
PREDICT_TIME="00:45:00"
PREDICT_EXTRA_FLAGS=""

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
