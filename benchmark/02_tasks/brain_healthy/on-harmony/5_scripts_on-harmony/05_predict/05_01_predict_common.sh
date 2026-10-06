#!/usr/bin/env bash
# Shared predict template for on-harmony -- sourced by 05_0X_predict_<method>_<contrast>.sh,
# NOT run directly. Thin shim: sources env.sh, sets on-harmony's config, delegates to the
# shared driver benchmark/00_commun_scripts/00_02_predict/predict_common.sh, adding one
# on-harmony-specific post-step through the driver's PREDICT_POST_ITEM_CMD hook: resampling
# RAS-space predictions back to native geometry inside each predict job (see
# 05_02_resample_predictions_to_native.py's docstring for why this dataset alone needs it).
#
# Cross-contrast test set: ALL 6 held-out contrasts (T1w/T2w/bold/dwi_ap/epi_ap/
# gre_echo1_mag) are predicted regardless of which contrast trained the model -- run
# 05_00_build_test_inputs.sh first (idempotent, builds the shared test set once).
#
# PREDICT_DATASET_ID_DEFAULT is fixed at 031 (Dataset031_OnHarmonyT1w31) REGARDLESS of
# which per-contrast Dataset actually trained the model -- this is the id whose
# imagesTs_<contrast>/ holds the test inputs (05_00_build_test_inputs.py's
# PREDICT_TARGET_DATASET), NOT the id the model was trained under. PREDICT_MODEL_DATASET_ID
# is set separately per TRAINING_CONTRAST for the checkpoint lookup (nnU-Net resolves
# nnUNet_results/Dataset<-d>_.../<trainer>__... using -d) -- same split, same bug class
# documented in predict_common.sh/totalseg-pelvic's shim (2026-09-16): without it every
# T2w/dwi_ap-trained prediction would silently produce zero output files.
#
# ⚠️ KNOWN LAYOUT CHANGE vs. the old inline-predict script: this dataset's historical
# default checkpoint is checkpoint_final.pth (NOT checkpoint_best.pth, unlike every other
# dataset), and the OLD inline script wrote checkpoint_final predictions to a FLAT
# fold{F}/<contrast>/ path (subdir only for non-default checkpoints). predict_common.sh's
# shared convention is the opposite (flat only for checkpoint_best, subdir for everything
# else, including "final") -- it cannot be changed here without affecting every other
# dataset. Consequence: NEW predictions run through this shim land at
# fold{F}/final/<contrast>/ (nested), not the old flat fold{F}/<contrast>/. Existing
# on-disk results from the old inline script are untouched; the rewritten
# 06_01_evaluate_testset.sh reads from wherever THIS shim actually wrote (the new nested
# path), so nothing is silently mismatched -- just be aware the path changed for anyone
# writing a new one-off script against the old flat convention.
set -euo pipefail
cd "${PROJECT_ROOT:?source 00_utils/env.sh (or env_t2w.sh/env_dwi.sh) BEFORE this shim}"

PREDICT_MODE="own"
PREDICT_JOB_PREFIX="onharmony_predict"
PREDICT_LOG_PREFIX="predict"
PREDICT_ITEMS_DEFAULT="T1w T2w bold dwi_ap epi_ap gre_echo1_mag"
PREDICT_FOLD_DEFAULT="all"
PREDICT_DATASET_ID_DEFAULT="031"
case "${TRAINING_CONTRAST:-T1w}" in
    T1w)    PREDICT_MODEL_DATASET_ID="031" ;;
    T2w)    PREDICT_MODEL_DATASET_ID="032" ;;
    dwi_ap) PREDICT_MODEL_DATASET_ID="033" ;;
    *) echo "05_01_predict_common.sh: unknown TRAINING_CONTRAST '${TRAINING_CONTRAST:-}'" >&2; exit 1 ;;
esac
PREDICT_TIME="01:00:00"
PREDICT_EXTRA_FLAGS="--disable_tta"
# Preserve this dataset's historical default checkpoint (see header note above) --
# predict_common.sh's own default is checkpoint_best.pth, wrong for on-harmony unless a
# caller explicitly wants the best-checkpoint arm.
CHECKPOINT="${CHECKPOINT:-checkpoint_final.pth}"

# on-harmony-only post-step: resample RAS predictions back to native geometry. Run INSIDE each
# predict job through the shared driver's PREDICT_POST_ITEM_CMD hook (2026-10-06). It used to run
# here, after the driver returned -- a silent no-op in node-pack record mode (no prediction
# exists yet at record time), which got x-mirrored predictions of the natively-LAS contrasts
# scored once (2026-10-05). 06_01_evaluate_testset.sh now also refuses off-grid predictions.
_TESTSET="${PREDICTIONS_ROOT}/${MODEL_TYPE}/_test_set"
PREDICT_POST_ITEM_CMD=".venv/bin/python '${PROJECT_ROOT}/benchmark/02_tasks/brain_healthy/on-harmony/5_scripts_on-harmony/05_predict/05_02_resample_predictions_to_native.py' '{out}' '${_TESTSET}/{item}/images_native'"

source "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
