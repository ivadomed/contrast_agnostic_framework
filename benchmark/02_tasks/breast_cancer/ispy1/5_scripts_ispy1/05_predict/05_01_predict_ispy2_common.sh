#!/usr/bin/env bash
# ============================================================================
#  ispy1 prediction — USES MODELS TRAINED ON ANOTHER DATASET (ispy2).
#  ispy1 is EVALUATION-ONLY (167 MAMA-MIA-expert-masked I-SPY1 cases, see
#  00_utils/env.sh). Same shape as duke-breast-mri's 05_01_predict_ispy2_common.sh;
#  both test items (t1wce, precontrast) predicted in ONE job per fold.
# ============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

PREDICT_MODE="cross"
SOURCE_PREFIX="ISPY2"
PREDICT_JOB_PREFIX="ispy1_predict"
PREDICT_LOG_PREFIX="ispy1_predict"
PREDICT_ITEMS_DEFAULT="t1wce precontrast"
PREDICT_FOLD_DEFAULT="all"
PREDICT_TIME="01:00:00"
PREDICT_EXTRA_FLAGS="-npp 4 -nps 2"

source "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
