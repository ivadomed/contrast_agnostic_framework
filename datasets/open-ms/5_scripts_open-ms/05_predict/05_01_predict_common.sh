#!/usr/bin/env bash
# Shared predict template for open-ms — sourced by 05_0X_predict_<method>.sh, NOT run
# directly. Sets the open-ms own-model config and delegates to the shared driver
# datasets/00_commun_scripts/00_02_predict/predict_common.sh. Predicts each held-out test
# patient on FLAIR (in-domain) + T2W + T1W (cross-contrast), all 4 folds.
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"

PREDICT_MODE="own"
PREDICT_JOB_PREFIX="openms_predict"
PREDICT_LOG_PREFIX="predict"
PREDICT_ITEMS_DEFAULT="flair t2w t1w"     # imagesTs_<item>/ dirs built by 02_00_convert.py
PREDICT_FOLD_DEFAULT="all"
PREDICT_DATASET_ID_DEFAULT="70"
PREDICT_TIME="00:45:00"    # inference is minutes, not hours — don't inherit the
                           # 60h training-time default (env.sh's RUN_JOB_TIME_DEFAULT),
                           # a long walltime request tanks Slurm backfill priority
PREDICT_EXTRA_FLAGS=""

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
