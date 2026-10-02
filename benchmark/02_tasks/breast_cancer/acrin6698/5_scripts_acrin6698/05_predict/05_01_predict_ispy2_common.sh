#!/usr/bin/env bash
# ============================================================================
#  acrin6698 prediction — USES MODELS TRAINED ON ANOTHER DATASET (ispy2).
#  acrin6698 is EVALUATION-ONLY (ACRIN-6698 T0 DWI, see 00_utils/env.sh). Same
#  shape as duke-breast-mri's 05_01_predict_ispy2_common.sh; single item `dwi_uniap` (L-R + skin-anchored A-P crop).
# ============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

PREDICT_MODE="cross"
SOURCE_PREFIX="ISPY2"
PREDICT_JOB_PREFIX="acrin6698_predict"
PREDICT_LOG_PREFIX="acrin6698_predict"
PREDICT_ITEMS_DEFAULT="dwi_uniap"   # 2026-10-01: skin-anchored A-P crop (02_02); plain `dwi` = L-R-only crop, superseded
PREDICT_FOLD_DEFAULT="all"
PREDICT_TIME="01:00:00"
PREDICT_EXTRA_FLAGS="-npp 4 -nps 2"

source "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
