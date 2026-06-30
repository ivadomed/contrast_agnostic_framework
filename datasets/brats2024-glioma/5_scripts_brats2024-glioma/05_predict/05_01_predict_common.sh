#!/usr/bin/env bash
# Shared predict template for brats2024-glioma — sourced by 05_0X_predict_<method>.sh,
# NOT run directly. Thin shim: sources env.sh, sets the brats config, and delegates to
# the shared driver datasets/00_commun_scripts/00_02_predict/predict_common.sh.
# Predicts with brats's OWN trained models. Test-input dirs (imagesTs_<contrast>/) must
# exist — build once with: python 05_00_build_test_inputs.py. See the driver header for
# the per-method env vars and positional args (RUN_ID / FOLD / contrasts...).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

PREDICT_MODE="own"
PREDICT_JOB_PREFIX="brats_predict"
PREDICT_LOG_PREFIX="predict"
PREDICT_ITEMS_DEFAULT="t1n t1c t2w t2f"
PREDICT_FOLD_DEFAULT="all"        # standardized: predict all 4 folds by default (matches eval + cross datasets)
PREDICT_DATASET_ID_DEFAULT="051"
PREDICT_TIME="0:45:00"
PREDICT_EXTRA_FLAGS=""

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
