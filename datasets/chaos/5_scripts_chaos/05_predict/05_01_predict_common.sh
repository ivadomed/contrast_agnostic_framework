#!/usr/bin/env bash
# Shared predict template for chaos — sourced by 05_0X_predict_<method>.sh, NOT run
# directly. Thin shim: sources env.sh, sets the chaos config, and delegates to the
# shared driver datasets/00_commun_scripts/00_02_predict/predict_common.sh.
# Predicts with chaos's OWN trained models over the per-modality CHAOS test set
# (imagesTs_<mod>/ built by 05_00_build_test_inputs.py). See the driver header for the
# per-method env vars and positional args (RUN_ID / FOLD / modalities...).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

PREDICT_MODE="own"
PREDICT_JOB_PREFIX="chaos_predict"
PREDICT_LOG_PREFIX="predict"
PREDICT_ITEMS_DEFAULT="t1in t1out t2spir ct"
PREDICT_FOLD_DEFAULT="all"        # standardized: predict all 4 folds by default (matches eval + cross datasets)
PREDICT_DATASET_ID_DEFAULT="60"
PREDICT_TIME=""                 # chaos omits --time (uses run_job's default)
PREDICT_EXTRA_FLAGS=""

# ── Translation-robustness experiment (opt-in via EXP_TRANSLATION=NNN) ────────
# When set (e.g. EXP_TRANSLATION=050), predict on the translated test variant
# imagesTs_<mod>_translation_NNN (built by 03_02_make_translated_test.py) and write
# predictions under .../fold{F}/exp_translation_NNN/<mod>/. Unset → normal predict.
if [ -n "${EXP_TRANSLATION:-}" ]; then
    PREDICT_INPUT_SUFFIX="_translation_${EXP_TRANSLATION}"
    PREDICT_OUTPUT_SUBDIR="exp_translation_${EXP_TRANSLATION}"
    PREDICT_JOB_PREFIX="chaos_predict_xl${EXP_TRANSLATION}"
fi

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
