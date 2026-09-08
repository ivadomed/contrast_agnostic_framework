#!/usr/bin/env bash
# ============================================================================
#  toothfairy2 prediction — OWN trained models on toothfairy2's OWN held-out
#  CBCT test cases (the 71 sealed cases in 4_splits_toothfairy2/test_cases.json).
#
#  NOTE ON WHAT THIS DOES *NOT* COVER. Every other training dataset in this
#  project gets its cross-contrast OOD axis from a SECOND in-house training
#  modality; toothfairy2 has only one (CBCT), so this driver produces the
#  IN-DOMAIN half of the picture only. The cross-MODALITY half — the headline
#  result for a domain-randomization method — comes from the external hanseg
#  set (head-and-neck CT + MR-T1, mandible only) via
#  datasets/hanseg/5_scripts_hanseg/. Do not read this driver's numbers as the
#  generalization result; they are the in-domain reference it is measured against.
# ============================================================================
# Thin shim: sources env.sh, sets the toothfairy2 own-model config, delegates to
# datasets/00_commun_scripts/00_02_predict/predict_common.sh. Sourced by the
# 05_XX_predict_<method>.sh wrappers (which set METHOD/TRAINER/CATEGORY);
# RUN_ID is a positional arg with no default — this is OWN mode.
set -euo pipefail
cd "${PROJECT_ROOT:?source 00_utils/env.sh BEFORE this shim}"

PREDICT_MODE="own"
PREDICT_JOB_PREFIX="tf2_predict_own"
PREDICT_LOG_PREFIX="tf2_predict_own"
PREDICT_ITEMS_DEFAULT="cbct"
PREDICT_FOLD_DEFAULT="all"
PREDICT_TIME="01:00:00"
PREDICT_EXTRA_FLAGS="-npp 4 -nps 2"

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
