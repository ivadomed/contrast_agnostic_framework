#!/usr/bin/env bash
# ============================================================================
#  duke-breast-mri prediction — USES MODELS TRAINED ON ANOTHER DATASET (ispy2).
#  duke-breast-mri is EVALUATION-ONLY (291 MAMA-MIA-expert-masked Duke-Breast-
#  Cancer-MRI cases, t1wce only -- see 00_utils/env.sh and
#  02_nnunet/02_01_convert_test_t1wce.py). Modeled directly on ambl's
#  05_25_predict_ispy2_common.sh.
# ============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

PREDICT_MODE="cross"
SOURCE_PREFIX="ISPY2"
PREDICT_JOB_PREFIX="duke_predict"
PREDICT_LOG_PREFIX="duke_predict"
# Duke has only ONE contrast available (post-contrast T1w DCE) -- both the
# ispy2 t1wce-trained AND t2w-trained models are tested on this single item
# (same-modality-family for the t1wce-trained model, cross-contrast for the
# t2w-trained one), per the project's "train on one modality -> predict/eval
# on every available modality of the target dataset" rule.
PREDICT_ITEMS_DEFAULT="t1wce"
PREDICT_FOLD_DEFAULT="all"
PREDICT_TIME="00:30:00"
PREDICT_EXTRA_FLAGS="-npp 4 -nps 2"

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
