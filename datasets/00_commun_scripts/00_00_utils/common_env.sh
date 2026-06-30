#!/usr/bin/env bash
# Shared per-dataset environment loader for the standard 9-subdir dataset layout.
#
# A dataset's 5_scripts_<name>/00_utils/env.sh acts as the dataset "config": it sets
# a handful of variables, then sources THIS file, which derives every standard path
# from $DATASET_NAME (so e.g. 2_nnUNet_<name>/, 4_splits_<name>/, 8_results_<name>/
# are spelled out once here instead of being re-hardcoded in every dataset).
#
# The caller MUST set before sourcing:
#   DATASET_ROOT   absolute path to datasets/<name>   (the caller computes this from
#                  its OWN ${BASH_SOURCE[0]} — it cannot be derived here, because here
#                  BASH_SOURCE points at common_env.sh, not the dataset's env.sh).
#   DATASET_NAME   dataset slug, e.g. "chaos" — must match the <name> in every subdir.
#
# The caller MAY set before sourcing:
#   DATASET_ROLE         training | test-only | both                 (default: training)
#   BIDS_SUBDIR          leaf dir under 1_BIDS_<name>/  → exports BIDS_ROOT if set
#   CE_SUBDIRS           space-separated subset of {raw preprocessed splits}: which
#                        optional standard subdirs this dataset actually has. Each maps
#                        to RAW_ROOT / nnUNet_preprocessed / SPLITS_DIR respectively.
#   CE_EXTRA_PYTHONPATH  extra dir(s) to insert on PYTHONPATH right after SCRIPTS_DIR
#                        (used by cross-dataset/test-only datasets to add e.g. the chaos
#                        scripts dir so chaos trainer classes resolve at predict time).
#
# common_env exports: PROJECT_ROOT, DATASET_ROLE, nnUNet_raw, PREDICTIONS_ROOT,
#   METRICS_ROOT, WANDB_PROJECT, PYTHONPATH, plus (conditionally) BIDS_ROOT, RAW_ROOT,
#   nnUNet_preprocessed, SPLITS_DIR. It also sources run_job.sh. SCRIPTS_DIR is set as
#   a plain (non-exported) shell var, matching the original env.sh behaviour.
#
# Dataset-specific vars stay in the caller's env.sh (set either before or after sourcing
# this, as the original ordering requires): NNUNET_DATASET_ID, MODEL_TYPE,
# TRAINING_CONTRAST + its default, the nnUNet_results default, and the CHAOS_* / other
# cross-dataset blocks.

: "${DATASET_ROOT:?common_env.sh: DATASET_ROOT must be set before sourcing}"
: "${DATASET_NAME:?common_env.sh: DATASET_NAME must be set before sourcing}"

export PROJECT_ROOT="$(cd "${DATASET_ROOT}/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/job_runner/run_job.sh"

export DATASET_ROLE="${DATASET_ROLE:-training}"

export nnUNet_raw="${DATASET_ROOT}/2_nnUNet_${DATASET_NAME}/raw"
export PREDICTIONS_ROOT="${DATASET_ROOT}/8_results_${DATASET_NAME}/01_predictions"
export METRICS_ROOT="${DATASET_ROOT}/8_results_${DATASET_NAME}/02_metrics"
export WANDB_PROJECT="mri_synthesis_seg_${DATASET_NAME}"

[ -n "${BIDS_SUBDIR:-}" ] && \
    export BIDS_ROOT="${DATASET_ROOT}/1_BIDS_${DATASET_NAME}/${BIDS_SUBDIR}"

for _d in ${CE_SUBDIRS:-}; do
    case "$_d" in
        raw)          export RAW_ROOT="${DATASET_ROOT}/0_raw_${DATASET_NAME}";;
        preprocessed) export nnUNet_preprocessed="${DATASET_ROOT}/2_nnUNet_${DATASET_NAME}/preprocessed";;
        splits)       export SPLITS_DIR="${DATASET_ROOT}/4_splits_${DATASET_NAME}";;
        *) echo "common_env.sh: unknown CE_SUBDIRS entry '$_d' (want: raw|preprocessed|splits)" >&2;;
    esac
done

SCRIPTS_DIR="${DATASET_ROOT}/5_scripts_${DATASET_NAME}"
export PYTHONPATH="${SCRIPTS_DIR}${CE_EXTRA_PYTHONPATH:+:${CE_EXTRA_PYTHONPATH}}:${PYTHONPATH:-}"
