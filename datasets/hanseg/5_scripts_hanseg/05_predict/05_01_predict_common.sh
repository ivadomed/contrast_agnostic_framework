#!/usr/bin/env bash
# ============================================================================
#  HaN-Seg prediction — USES MODELS TRAINED ON toothfairy2 (CBCT).
#  hanseg is EVALUATION-ONLY. These load toothfairy2-trained checkpoints and run
#  them over the FOV-matched head-and-neck cases, measuring cross-MODALITY
#  generalization CBCT -> conventional CT (01_01_prepare_ct.py) and
#  CBCT -> T1 MRI (01_02_prepare_mr.py) on the mandible.
#
#  This is the OOD half of the toothfairy2 task. toothfairy2 has one training
#  modality, so its own held-out CBCT test set is entirely in-domain — the
#  generalization claim rests on THIS driver's outputs, not that one's.
#
#  Model dir + nnUNet_raw/preprocessed come from the TF2_* env vars (set by
#  00_utils/env.sh, redirected to scratch by scripts/cluster/tamia_env_hanseg.sh);
#  outputs land under PREDICTIONS_ROOT/toothfairy2_model/cbct/<category>/.
# ============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

PREDICT_MODE="cross"
SOURCE_PREFIX="TF2"
PREDICT_JOB_PREFIX="hanseg_predict"
PREDICT_LOG_PREFIX="hanseg_predict"
# Two items, both scored against the SAME original labelsTs_ct:
#   ct    — the CT image
#   mrt1  — the MR image resampled into the CT frame (01_04_prepare_mr_in_ct_frame.py)
# No label is ever transformed. This replaces the earlier label-propagation attempt
# (01_02_prepare_mr.py, disabled) and follows what the HaN-Seg challenge teams did:
# register MR->CT and keep the ground truth in CT space.
PREDICT_ITEMS_DEFAULT="ct mrt1"
PREDICT_FOLD_DEFAULT="all"
PREDICT_TIME="00:45:00"
PREDICT_EXTRA_FLAGS="-npp 4 -nps 2"

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
