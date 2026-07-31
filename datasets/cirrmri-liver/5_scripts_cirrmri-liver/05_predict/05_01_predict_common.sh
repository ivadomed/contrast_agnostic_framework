#!/usr/bin/env bash
# ============================================================================
#  CIRRMRI-LIVER prediction — USES MODELS TRAINED ON ANOTHER DATASET (chaos).
#  CIRRMRI-LIVER is EVALUATION-ONLY (see datasets/cirrmri-liver/README.md): no
#  native models. These load chaos-trained checkpoints (MR T1-DUAL in-phase /
#  T2-SPIR) and run them over CIRRMRI-LIVER's T1w+T2w volumes to measure MRI->MRI
#  (different scanner/protocol/pathology cohort) generalization. Model dir +
#  nnUNet_raw/preprocessed come from the CHAOS_* env vars; outputs are segregated
#  under PREDICTIONS_ROOT/<chaos_model_type>/<contrast>/ so any future native
#  predictions live elsewhere.
# ============================================================================
# Thin shim: sources env.sh, sets the cirrmri-liver cross-dataset config, and
# delegates to the shared driver datasets/00_commun_scripts/00_02_predict/predict_common.sh.
# Sourced by 05_0X_predict_chaos_<method>.sh (which sets METHOD/TRAINER/CATEGORY/RUN_ID).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

PREDICT_MODE="cross"
PREDICT_JOB_PREFIX="cirrmri_liver_predict"
PREDICT_LOG_PREFIX="cirrmri_liver_predict"
PREDICT_ITEMS_DEFAULT="t1 t2"
PREDICT_FOLD_DEFAULT="all"
PREDICT_TIME="01:30:00"
PREDICT_EXTRA_FLAGS="-npp 12 -nps 6"

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
