#!/usr/bin/env bash
# ============================================================================
#  LiverHccSeg prediction — USES MODELS TRAINED ON ANOTHER DATASET (atlas-liver-hcc).
#  LiverHccSeg is EVALUATION-ONLY (see datasets/liverhccseg/README.md): no native
#  models. These load atlas-liver-hcc-trained checkpoints (T1w CE-MRI) and run them
#  over LiverHccSeg's 4-phase CE-T1w volumes (independent TCGA-LIHC cohort) to add a
#  same-disease, same-contrast-family, different-cohort stratum alongside
#  lld-mmri-hcc's cross-contrast one. Model dir + nnUNet_raw/preprocessed come from
#  the ATLAS_* env vars; outputs segregated under PREDICTIONS_ROOT/<atlas_model_type>/<contrast>/.
# ============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

PREDICT_MODE="cross"
SOURCE_PREFIX="ATLAS"
PREDICT_JOB_PREFIX="liverhccseg_predict"
PREDICT_LOG_PREFIX="liverhccseg_predict"
PREDICT_ITEMS_DEFAULT="ce-pre_T1w ce-art_T1w ce-pv_T1w ce-del_T1w"
PREDICT_FOLD_DEFAULT="all"
PREDICT_TIME="00:45:00"
PREDICT_EXTRA_FLAGS="-npp 6 -nps 3"

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
