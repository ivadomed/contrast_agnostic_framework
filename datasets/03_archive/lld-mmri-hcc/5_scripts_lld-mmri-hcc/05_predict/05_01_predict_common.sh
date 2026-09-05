#!/usr/bin/env bash
# ============================================================================
#  LLD-MMRI-HCC prediction — USES MODELS TRAINED ON ANOTHER DATASET (atlas-liver-hcc).
#  LLD-MMRI-HCC is EVALUATION-ONLY (see datasets/lld-mmri-hcc/README.md): no native
#  models. These load atlas-liver-hcc-trained checkpoints (T1w CE-MRI) and run them
#  over LLD-MMRI-HCC's T2WI/DWI volumes to measure cross-contrast generalization on
#  the tumour/lesion label. Model dir + nnUNet_raw/preprocessed come from the ATLAS_*
#  env vars; outputs are segregated under PREDICTIONS_ROOT/<atlas_model_type>/<contrast>/
#  so any future native predictions live elsewhere.
# ============================================================================
# Thin shim: sources env.sh, sets the lld-mmri-hcc cross-dataset config, and delegates
# to the shared driver datasets/00_commun_scripts/00_02_predict/predict_common.sh.
# Sourced by 05_0X_predict_atlas_<method>.sh (which sets METHOD/TRAINER/CATEGORY/RUN_ID).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

PREDICT_MODE="cross"
SOURCE_PREFIX="ATLAS"
PREDICT_JOB_PREFIX="lldmmrihcc_predict"
PREDICT_LOG_PREFIX="lldmmrihcc_predict"
PREDICT_ITEMS_DEFAULT="t2wi dwi"
PREDICT_FOLD_DEFAULT="all"
PREDICT_TIME="00:45:00"
PREDICT_EXTRA_FLAGS="-npp 6 -nps 3"

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
