#!/usr/bin/env bash
# ============================================================================
#  I-SPY2 prediction — OWN trained models on I-SPY2's OWN held-out test cases.
#  This is the headline ispy2-internal cross-contrast eval (2026-09-04/05
#  pivot): I-SPY2 is now the primary breast-cancer TRAINING dataset (see
#  CLAUDE.md/00_utils/env.sh's DATASET_ROLE="both"). A t1wce- (or t2w-) trained
#  model predicts on I-SPY2's own held-out test cases for BOTH modalities
#  (see 4_splits_ispy2/splits_final.json + test_cases.json) -- matched-contrast
#  in-domain AND cross-contrast OOD, same pattern as chaos/brats2024-glioma/
#  ambl's own predict_common.sh. Inputs read from nnUNet_raw/Dataset<id>_.../
#  imagesTs_<item>/ -- on tamia these are symlinks into the flat top-level
#  imagesTs_<item>/ dirs 05_00_build_test_inputs.py wrote (see that script +
#  scripts/cluster/tamia_env_ispy2.sh); locally they are already nested.
# ============================================================================
# Thin shim: sources env.sh (or env_t2w.sh), sets the ispy2 own-model config,
# and delegates to the shared driver
# datasets/00_commun_scripts/00_02_predict/predict_common.sh. Sourced by
# 05_2X_predict_own_<t1wce|t2w>_<method>.sh (sets METHOD/TRAINER/CATEGORY/
# DATASET_ID; RUN_ID is a positional arg, no default -- this is OWN mode).
set -euo pipefail
cd "${PROJECT_ROOT:?source 00_utils/env.sh (or env_t2w.sh) BEFORE this shim}"

PREDICT_MODE="own"
PREDICT_JOB_PREFIX="ispy2_predict_own"
PREDICT_LOG_PREFIX="ispy2_predict_own"
PREDICT_ITEMS_DEFAULT="t1wce t2w"
PREDICT_FOLD_DEFAULT="all"
PREDICT_TIME="00:30:00"
PREDICT_EXTRA_FLAGS="-npp 4 -nps 2"

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
