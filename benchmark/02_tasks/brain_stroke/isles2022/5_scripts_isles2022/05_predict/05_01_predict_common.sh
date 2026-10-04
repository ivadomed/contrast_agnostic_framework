#!/usr/bin/env bash
# Shared predict shim for isles2022 — sourced by 05_XX_predict_<contrast>_<method>.sh, NOT run directly. Sets the own-model config and
# delegates to the shared driver benchmark/00_commun_scripts/00_02_predict/predict_common.sh. Predicts every held-out test
# case on dwi (+adc +flair), all 3 folds in parallel. The wrapper sets METHOD/TRAINER/CATEGORY/DATASET_ID (and has already
# sourced env.sh or env_flair.sh, so TRAINING_CONTRAST/nnUNet_results match the model being predicted).
set -euo pipefail
cd "${PROJECT_ROOT:?source 00_utils/env.sh (or env_flair.sh) BEFORE this shim}"

PREDICT_MODE="own"
PREDICT_JOB_PREFIX="isles2022_predict"
PREDICT_LOG_PREFIX="isles2022_predict"
PREDICT_ITEMS_DEFAULT="dwi adc flair"      # imagesTs_<item>/ dirs built by 02_nnunet/02_01_convert.py
PREDICT_FOLD_DEFAULT="all"
PREDICT_TIME="${PREDICT_TIME:-00:45:00}"   # inference is minutes; don't inherit the 48h training default
PREDICT_EXTRA_FLAGS="-npp 4 -nps 2"        # fewer preprocessing/export workers (blosc2/thread-limit gotcha)

source "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
