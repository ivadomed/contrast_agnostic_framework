#!/usr/bin/env bash
# KIDNEY-T2W prediction — USES MODELS TRAINED ON ANOTHER DATASET (chaos).
# KIDNEY-T2W is EVALUATION-ONLY (see datasets/kidney-t2w/README.md): no native
# models. These load chaos-trained checkpoints (MR T1-DUAL in-phase / T2-SPIR)
# and run them over KIDNEY-T2W's T2-weighted volumes to measure MRI->MRI
# generalization on the kidney label.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

PREDICT_MODE="cross"
PREDICT_JOB_PREFIX="kidney_t2w_predict"
PREDICT_LOG_PREFIX="kidney_t2w_predict"
PREDICT_ITEMS_DEFAULT="t2"
PREDICT_FOLD_DEFAULT="all"
PREDICT_TIME="01:00:00"
PREDICT_EXTRA_FLAGS="-npp 6 -nps 3"

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
