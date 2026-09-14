#!/usr/bin/env bash
# ============================================================================
#  I-SPY2 prediction — USES MODELS TRAINED ON ANOTHER DATASET (ambl).
#  I-SPY2 is EVALUATION-ONLY (see datasets/ispy2/1_BIDS_ispy2/breast-ispy2/README):
#  no native models. These load ambl-trained checkpoints (breast DCE-MRI, T1WCE or
#  T2w depending on the wrapper) and run them over I-SPY2's 122 usable BILATERAL
#  cases (see 00_utils/env.sh + 05_00_build_test_inputs.py — unilateral cases are
#  never fed to prediction) to measure cross-DATASET generalization on the tumour/
#  lesion label. Model dir + nnUNet_raw/preprocessed come from the AMBL_* env vars
#  (set by 00_utils/env.sh, overridden to scratch by scripts/cluster/tamia_env_ispy2.sh);
#  outputs are segregated under PREDICTIONS_ROOT/<ambl_model_type>/<contrast>/.
# ============================================================================
# Thin shim: sources env.sh, sets the ispy2 cross-dataset config, and delegates
# to the shared driver datasets/00_commun_scripts/00_02_predict/predict_common.sh.
# Sourced by 05_0X_predict_ambl_<t1wce|t2w>_<method>.sh (which sets
# METHOD/TRAINER/CATEGORY/AMBL_TRAINING_CONTRAST/AMBL_DATASET_ID/RUN_ID).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

PREDICT_MODE="cross"
SOURCE_PREFIX="AMBL"
PREDICT_JOB_PREFIX="ispy2_predict"
PREDICT_LOG_PREFIX="ispy2_predict"
# Both items, matching ambl's own train-on-one/predict-on-all-modalities pattern:
# a t1wce-trained model is tested on ispy2's t1wce (matched contrast, still cross-
# DATASET) AND t2w (cross-contrast + cross-dataset) items, and vice versa.
PREDICT_ITEMS_DEFAULT="t1wce t2w"
PREDICT_FOLD_DEFAULT="all"
PREDICT_TIME="00:30:00"
PREDICT_EXTRA_FLAGS="-npp 6 -nps 3"

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
