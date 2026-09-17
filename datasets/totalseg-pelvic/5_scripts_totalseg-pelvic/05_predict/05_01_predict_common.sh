#!/usr/bin/env bash
# Shared predict template for totalseg-pelvic — sourced by 05_0X_predict_<method>[_mri].sh,
# NOT run directly. Thin shim: sources env.sh, sets the totalseg-pelvic config, and
# delegates to the shared driver datasets/00_commun_scripts/00_02_predict/predict_common.sh.
#
# Predicts with totalseg-pelvic's OWN trained models (CT- or MRI-trained, whichever
# wrapper set TRAINER/DATASET_ID) over TWO items, both consolidated under Dataset130's
# raw tree by 05_00_build_test_inputs.py:
#   ct   — held-out CT test cases   (own-contrast for a CT-trained model, cross-contrast for MRI-trained)
#   mri  — held-out MRI test cases  (mirror of the above)
#
# PREDICT_DATASET_ID_DEFAULT is fixed at 130 REGARDLESS of which per-modality Dataset
# actually trained the model — this is the id whose imagesTs_*/labelsTs_* holds the
# test inputs, see 05_00_build_test_inputs.py's docstring — NOT the id the model was
# trained under.
#
# PREDICT_MODEL_DATASET_ID must be set separately for the checkpoint lookup (nnU-Net
# resolves nnUNet_results/Dataset<-d>_.../<trainer>__... using -d, which is 131 for an
# MRI-trained model, 130 for CT). Bug found 2026-09-16 (same root cause independently
# hit autopet's PET-trained methods the same day, see predict_common.sh's own comment):
# without this split every MRI-trained prediction silently produced zero output files —
# nnUNetv2_predict's FileNotFoundError on a nonexistent Dataset130_.../dataset.json went
# only to the per-fold pack log, never surfaced by predict_common.sh, which prints
# "done" for every fold regardless of whether a case actually got written.
set -euo pipefail
cd "${PROJECT_ROOT:?source 00_utils/env.sh (or env_mri.sh) BEFORE this shim}"

PREDICT_MODE="own"
PREDICT_JOB_PREFIX="totalseg_pelvic_predict"
PREDICT_LOG_PREFIX="predict"
PREDICT_ITEMS_DEFAULT="ct mri"
PREDICT_FOLD_DEFAULT="all"
PREDICT_DATASET_ID_DEFAULT="130"
PREDICT_MODEL_DATASET_ID="130"
[ "${TRAINING_CONTRAST:-ct}" = "mri" ] && PREDICT_MODEL_DATASET_ID="131"
PREDICT_TIME="01:00:00"
PREDICT_EXTRA_FLAGS="-npp 4 -nps 2"

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
