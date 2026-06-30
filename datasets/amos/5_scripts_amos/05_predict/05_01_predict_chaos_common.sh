#!/usr/bin/env bash
# ============================================================================
#  AMOS prediction — USES MODELS TRAINED ON ANOTHER DATASET (chaos).
#  AMOS has no native models yet (training pending — see datasets/amos/README.md);
#  these load chaos-trained checkpoints (MR T1-DUAL in-phase) and run them over AMOS
#  CT and MRI as a cross-dataset generalization probe. Model dir + nnUNet_raw/preprocessed
#  come from the CHAOS_* env vars; outputs are segregated under
#  PREDICTIONS_ROOT/<chaos_model_type>/<contrast>/ so future AMOS-native predictions
#  never collide. When AMOS is trained natively, add a sibling PREDICT_MODE=own shim.
# ============================================================================
# Thin shim: sources env.sh, sets the amos cross-dataset config, and delegates to the
# shared driver datasets/00_commun_scripts/00_02_predict/predict_common.sh. Sourced by
# 05_0X_predict_chaos_<method>.sh (which sets METHOD/TRAINER/CATEGORY/RUN_ID). See the
# driver header for positional args (RUN_ID / FOLD / modalities...).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

PREDICT_MODE="cross"
PREDICT_JOB_PREFIX="amos_predict"
PREDICT_LOG_PREFIX="amos_predict"
PREDICT_ITEMS_DEFAULT="ct mri"
PREDICT_FOLD_DEFAULT="all"
PREDICT_TIME="00:45:00"
PREDICT_EXTRA_FLAGS="-npp 12 -nps 6"

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
