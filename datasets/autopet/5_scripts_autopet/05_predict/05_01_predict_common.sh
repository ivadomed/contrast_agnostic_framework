#!/usr/bin/env bash
# Shared predict template for autopet — sourced by 05_0X_predict_<method>[_pet].sh, NOT
# run directly. Thin shim: sources env.sh, sets the autopet config, and delegates to the
# shared driver datasets/00_commun_scripts/00_02_predict/predict_common.sh.
#
# Predicts with autopet's OWN trained models (CT- or PET-trained, whichever wrapper set
# TRAINER/DATASET_ID) over FOUR items, all consolidated under Dataset120's raw tree by
# 05_00_build_test_inputs.py / 05_00b_build_psma_test_inputs.py (same convention as
# chaos's own multi-item predict driver):
#   ct        — held-out FDG test cases, CT channel   (own-contrast for a CT-trained model,
#                                                        cross-contrast for a PET-trained one)
#   pet       — held-out FDG test cases, PET channel  (mirror of the above)
#   psma_ct   — PSMA cohort (cross-institution/cross-tracer, LMU), CT channel
#   psma_pet  — PSMA cohort, PET channel
#
# PREDICT_DATASET_ID_DEFAULT is fixed at 120 REGARDLESS of which per-modality Dataset
# actually trained the model — see 05_00_build_test_inputs.py's docstring for why (the
# driver looks up imagesTs_<item> under ONE dataset id, a purely organizational choice,
# not a claim about which Dataset trained the model).
set -euo pipefail
cd "${PROJECT_ROOT:?source 00_utils/env.sh (or env_pet.sh) BEFORE this shim}"

PREDICT_MODE="own"
PREDICT_JOB_PREFIX="autopet_predict"
PREDICT_LOG_PREFIX="predict"
PREDICT_ITEMS_DEFAULT="ct pet psma_ct psma_pet"
PREDICT_FOLD_DEFAULT="all"
PREDICT_DATASET_ID_DEFAULT="120"
PREDICT_TIME="02:00:00"   # whole-body volumes are large — more headroom than a typical 1h
PREDICT_EXTRA_FLAGS="-npp 4 -nps 2"

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
