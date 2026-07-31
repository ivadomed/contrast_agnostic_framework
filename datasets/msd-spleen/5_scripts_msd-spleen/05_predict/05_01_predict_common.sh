#!/usr/bin/env bash
# ============================================================================
#  MSD-SPLEEN prediction — USES MODELS TRAINED ON ANOTHER DATASET (chaos).
#  MSD-SPLEEN is EVALUATION-ONLY (see datasets/msd-spleen/README.md): no native models.
#  These load chaos-trained checkpoints (MR T1-DUAL in-phase / T2-SPIR) and run them
#  over MSD-SPLEEN's CT volumes to measure MR->CT generalization on the spleen label.
#  Model dir + nnUNet_raw/preprocessed come from the CHAOS_* env vars; outputs are
#  segregated under PREDICTIONS_ROOT/<chaos_model_type>/<contrast>/ so any future
#  native predictions live elsewhere. To predict with msd-spleen-native models, add
#  a PREDICT_MODE=own shim.
# ============================================================================
# Thin shim: sources env.sh, sets the msd-spleen cross-dataset config, and delegates
# to the shared driver datasets/00_commun_scripts/00_02_predict/predict_common.sh.
# Sourced by 05_0X_predict_chaos_<method>.sh (which sets METHOD/TRAINER/CATEGORY/RUN_ID).
# See the driver header for positional args (RUN_ID / FOLD / modalities...).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

PREDICT_MODE="cross"
PREDICT_JOB_PREFIX="msd_spleen_predict"
PREDICT_LOG_PREFIX="msd_spleen_predict"
PREDICT_ITEMS_DEFAULT="ct"
PREDICT_FOLD_DEFAULT="all"
PREDICT_TIME="00:45:00"
PREDICT_EXTRA_FLAGS="-npp 12 -nps 6"

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
