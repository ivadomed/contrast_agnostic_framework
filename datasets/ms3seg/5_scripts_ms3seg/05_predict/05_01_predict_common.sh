#!/usr/bin/env bash
# Thin shim: sources env.sh, sets the ms3seg cross-dataset config, delegates to the
# shared driver. Sourced by 05_0X_predict_openms_<method>.sh.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

SOURCE_PREFIX="OPENMS"
PREDICT_MODE="cross"
PREDICT_JOB_PREFIX="ms3seg_predict"
PREDICT_LOG_PREFIX="ms3seg_predict"
PREDICT_ITEMS_DEFAULT="flair t1w t2w"
PREDICT_FOLD_DEFAULT="all"
PREDICT_TIME="00:30:00"
PREDICT_EXTRA_FLAGS=""

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
