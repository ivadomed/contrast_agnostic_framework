#!/usr/bin/env bash
# ============================================================================
#  TRUSTED prediction — USES MODELS TRAINED ON ANOTHER DATASET (chaos).
#  TRUSTED is EVALUATION-ONLY (see datasets/trusted/README.md): no native models.
#  These load chaos-trained checkpoints (MR T1-DUAL in-phase / T2-SPIR) and run
#  them over TRUSTED's CT and US kidney volumes to measure MR→{CT,US} domain-
#  randomization generalization. Model dir + nnUNet_raw/preprocessed come from the
#  CHAOS_* env vars; outputs are segregated under PREDICTIONS_ROOT/<chaos_model_type>/
#  <contrast>/ so any future native predictions live elsewhere.
#
#  Unlike sliver07 (CT only) the default item list is "ct us" — two test modalities
#  fed sequentially within each fold's GPU job. Pass a subset as trailing args to
#  predict just one (e.g. `... <RUN_ID> all us`).
# ============================================================================
# Thin shim: sources env.sh, sets the trusted cross-dataset config, and delegates to the
# shared driver datasets/00_commun_scripts/00_02_predict/predict_common.sh. Sourced by
# 05_0X_predict_chaos_<method>.sh (which sets METHOD/TRAINER/CATEGORY/RUN_ID). See the
# driver header for positional args (RUN_ID / FOLD / items...).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

PREDICT_MODE="cross"
PREDICT_JOB_PREFIX="trusted_predict"
PREDICT_LOG_PREFIX="trusted_predict"
PREDICT_ITEMS_DEFAULT="ct us"
PREDICT_FOLD_DEFAULT="all"

# ct and us are submitted as SEPARATE jobs (the _all wrappers run ct and us passes
# independently). Both fit the fast defaults: CT volumes are small, and US is fed to
# nnU-Net pre-resampled to ~1.5 mm (05_00_build_test_inputs.py) so its prediction also
# lands on a small grid — no full-0.3 mm softmax export, so the 110 G/GPU default and
# the standard worker count are ample for both. (45 min per single-modality fold job.)
PREDICT_TIME="00:45:00"
PREDICT_EXTRA_FLAGS="-npp 12 -nps 6"

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
