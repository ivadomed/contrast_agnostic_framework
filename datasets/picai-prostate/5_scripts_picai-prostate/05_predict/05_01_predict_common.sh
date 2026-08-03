#!/usr/bin/env bash
# Shared predict template for picai-prostate — sourced by 05_XX_predict_<modality>_<method>.sh,
# NOT run directly. Sets the picai-prostate own-model config and delegates to the shared
# driver datasets/00_commun_scripts/00_02_predict/predict_common.sh. Predicts every held-out
# test study on T2W + ADC + HBV (in-domain + cross-contrast), folds 0 1 2.
#
# ADC wrappers pre-source ../00_utils/env_adc.sh and pre-export PREDICT_DATASET_ID_DEFAULT=81
# before sourcing this file; the `:-` guards below keep those.
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
[ -f "${PROJECT_ROOT}/scripts/cluster/tamia_env_picai.sh" ] && [ -d /scratch/p/paulh ] && \
    source "${PROJECT_ROOT}/scripts/cluster/tamia_env_picai.sh"

PREDICT_MODE="own"
PREDICT_JOB_PREFIX="picai_predict"
PREDICT_LOG_PREFIX="predict"
PREDICT_ITEMS_DEFAULT="t2w adc hbv"     # imagesTs_<item>/ dirs built by 02_00_convert.py
PREDICT_FOLD_DEFAULT="all"
PREDICT_DATASET_ID_DEFAULT="${PREDICT_DATASET_ID_DEFAULT:-80}"
# Inference is minutes, not hours — don't inherit env.sh's 60h training-time default, a long
# walltime request tanks Slurm backfill priority.
PREDICT_TIME="${PREDICT_TIME:-00:45:00}"
PREDICT_EXTRA_FLAGS=""

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
