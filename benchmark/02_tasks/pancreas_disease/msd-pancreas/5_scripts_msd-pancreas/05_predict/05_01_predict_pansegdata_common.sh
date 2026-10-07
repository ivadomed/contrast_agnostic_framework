#!/usr/bin/env bash
# Cross-dataset predict shim: pansegdata's trained models on msd-pancreas's test items. Sourced by the roster driver (05_02) per source run; sets the cross-mode config and hands off to
# the shared benchmark/00_commun_scripts/00_02_predict/predict_common.sh. METHOD/TRAINER/CATEGORY and PANSEG_TRAINING_CONTRAST/PANSEG_DATASET_ID are set by the driver.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
PREDICT_MODE="cross"
SOURCE_PREFIX="PANSEG"
PREDICT_JOB_PREFIX="msd-pancreas_predict"
PREDICT_LOG_PREFIX="msd-pancreas_predict"
PREDICT_ITEMS_DEFAULT="ct"
PREDICT_FOLD_DEFAULT="all"
PREDICT_TIME="03:00:00"   # 280 cases per fold job (the MRI companion had 109 at 01:00:00)
PREDICT_EXTRA_FLAGS="-npp 4 -nps 2"
source "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
