#!/usr/bin/env bash
# ============================================================================
#  MSLesSeg prediction — USES MODELS TRAINED ON ANOTHER DATASET (open-ms).
#  MSLesSeg is EVALUATION-ONLY (see datasets/mslesseg/README.md): no native models.
#  These load open-ms-trained checkpoints (brain MS lesion, FLAIR-trained) and run
#  them over ALL 115 MSLesSeg scans to measure MS-lesion domain-randomization
#  generalization to an independent cohort/scanner set. Model dir + nnUNet_raw/
#  preprocessed come from the OPENMS_* env vars (SOURCE_PREFIX=OPENMS); outputs are
#  segregated under PREDICTIONS_ROOT/<open_ms_model_type>/flair/ so any future native
#  predictions would live elsewhere.
#
#  Default item list is "flair t1w t2w" — open-ms's model is single-channel FLAIR-
#  trained, so all three MSLesSeg contrasts are fed as cross-contrast generalization
#  probes (same in-domain/cross-contrast split as open-ms's own held-out test).
# ============================================================================
# Thin shim: sources env.sh, sets the mslesseg cross-dataset config, and delegates to
# the shared driver datasets/00_commun_scripts/00_02_predict/predict_common.sh. Sourced
# by 05_0X_predict_openms_<method>.sh (which sets METHOD/TRAINER/CATEGORY/RUN_ID). See
# the driver header for positional args (RUN_ID / FOLD / items...).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

SOURCE_PREFIX="OPENMS"
PREDICT_MODE="cross"
PREDICT_JOB_PREFIX="mslesseg_predict"
PREDICT_LOG_PREFIX="mslesseg_predict"
PREDICT_ITEMS_DEFAULT="flair t1w t2w"
PREDICT_FOLD_DEFAULT="all"

# MSLesSeg volumes are small (182x218x182, ~7.2M voxels) — inference is minutes, not
# hours. Same rationale as trusted/open-ms's own PREDICT_TIME override (don't inherit
# a training-time default; a long walltime request tanks Slurm backfill priority).
PREDICT_TIME="00:30:00"
PREDICT_EXTRA_FLAGS=""

source "${PROJECT_ROOT}/datasets/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
