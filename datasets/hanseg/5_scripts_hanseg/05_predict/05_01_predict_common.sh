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
# `ct` ONLY. The mrt1 arm is built and working but is DISABLED by decision
# (2026-09-08) — see 01_prepare/01_02_prepare_mr.py's "STATUS" block. Its ground
# truth is propagated from CT by our own registration, and validation showed the QC
# gate used to accept it (tissue_frac) is ANTI-correlated with accuracy: a mask
# displaced 12 mm scores BETTER (0.967) than the correct one (0.943). The
# registration itself has support (edge_score peaks sharply at 0 displacement), but
# "GT we generated, validated by a metric we also designed" is not a defensible basis
# for a cross-modality claim in a paper.
# Re-enable deliberately with HANSEG_EVAL_ITEMS / by editing this line — do not turn
# it back on by accident.
PREDICT_ITEMS_DEFAULT="ct"
PREDICT_FOLD_DEFAULT="all"
PREDICT_TIME="00:45:00"
PREDICT_EXTRA_FLAGS="-npp 4 -nps 2"

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
