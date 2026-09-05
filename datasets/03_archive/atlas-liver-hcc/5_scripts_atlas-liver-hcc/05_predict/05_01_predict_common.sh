#!/usr/bin/env bash
# Shared predict template for atlas-liver-hcc — sourced by 05_0X_predict_<method>.sh,
# NOT run directly. Sets the atlas-liver-hcc own-model config and delegates to the
# shared driver datasets/00_commun_scripts/00_02_predict/predict_common.sh. Predicts
# each held-out test patient on T1w — the ONLY test item (single-modality dataset, no
# cross-contrast axis; see 00_utils/env.sh), all 3 folds.
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"

PREDICT_MODE="own"
PREDICT_JOB_PREFIX="atlashcc_predict"
PREDICT_LOG_PREFIX="predict"
PREDICT_ITEMS_DEFAULT="t1w"     # imagesTs_t1w/ dir built by 02_00_convert.py — single item
PREDICT_FOLD_DEFAULT="all"
PREDICT_DATASET_ID_DEFAULT="80"
PREDICT_TIME="${PREDICT_TIME:-00:45:00}"    # inference is minutes, not hours — don't
                           # inherit the 60h training-time default (env.sh's
                           # RUN_JOB_TIME_DEFAULT), a long walltime request tanks Slurm
                           # backfill priority. Overridable (e.g. PREDICT_TIME=00:30:00
                           # for a quick one-off) since it's a straight env passthrough.
PREDICT_EXTRA_FLAGS=""

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
