#!/usr/bin/env bash
# Source this at the top of every ispy1 pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#
# ispy1 = I-SPY 1 / ACRIN 6657 breast DCE-MRI (TCIA ISPY1) with MAMA-MIA expert
# primary-tumour masks (Synapse syn60868042) -- EVAL-ONLY cross-dataset test set
# for the I-SPY2-trained breast models (added 2026-09-30, third breast eval
# companion after ispy2's own test split and duke-breast-mri). Same MAMA-MIA
# annotation protocol as duke-breast-mri; natively unilateral sagittal 1.5T
# acquisitions (2002-2006), so it doubles as a scanner/era/orientation shift.
#
# Two test items, same 167 patients/masks: t1wce (1st post-contrast) and
# precontrast (native T1w) -- see 02_nnunet/02_01_convert_test.py for the
# pathology-match exclusions (bilateral cancer, implant) and size audit.
# License: CC BY 3.0. No DatasetXXX id (flat {imagesTs,labelsTs}_<item>/ layout,
# same eval-only convention as duke-breast-mri).

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET_NAME="ispy1"
export MODEL_TYPE="ispy1_model"
export DATASET_ROLE="eval_only"
export TRAINING_CONTRAST="${TRAINING_CONTRAST:-t1wce}"
export RUN_JOB_TIME_DEFAULT="${RUN_JOB_TIME_DEFAULT:-04:00:00}"

# (set BEFORE common_env.sh, which sources run_job and freezes its defaults)
# Small volumes (<=256x256x60) -- don't take the L40S defaults (16 CPU / 110G per GPU).
export RUN_JOB_CPUS_PER_GPU="${RUN_JOB_CPUS_PER_GPU:-8}"
export RUN_JOB_MEM_PER_GPU="${RUN_JOB_MEM_PER_GPU:-40G}"

export BIDS_SUBDIR="onc-breast-ispy1"
CE_SUBDIRS="preprocessed splits"
source "${DATASET_ROOT}/../../../00_commun_scripts/00_00_utils/common_env.sh"

export METRICS_ROOT="${METRICS_ROOT:-${DATASET_ROOT}/8_results_ispy1/02_metrics}"
export CHECKPOINTS_DIR="${DATASET_ROOT}/6_checkpoints_ispy1"
export RESULTS_DIR="${DATASET_ROOT}/8_results_ispy1"

# ── Cross-dataset model source: ispy2 (identical block to duke-breast-mri's) ──
export ISPY2_DATASET_ROOT="${ISPY2_DATASET_ROOT:-${DATASET_ROOT}/../ispy2}"
export ISPY2_PREDICTIONS_ROOT="${ISPY2_PREDICTIONS_ROOT:-${ISPY2_DATASET_ROOT}/8_results_ispy2/01_predictions}"
export ISPY2_NNUNET_RAW="${ISPY2_NNUNET_RAW:-${ISPY2_DATASET_ROOT}/2_nnUNet_ispy2/raw}"
export ISPY2_NNUNET_PREPROCESSED="${ISPY2_NNUNET_PREPROCESSED:-${ISPY2_DATASET_ROOT}/2_nnUNet_ispy2/preprocessed}"
export ISPY2_DATASET_ID="${ISPY2_DATASET_ID:-100}"
export ISPY2_TRAINING_CONTRAST="${ISPY2_TRAINING_CONTRAST:-t1wce}"
export ISPY2_MODEL_TYPE="ispy2_model"
# Identical label numbering to ispy2 (background=0, tumour=1) -- no --label_map needed.
export ISPY2_DATASET_JSON="${ISPY2_NNUNET_RAW}/Dataset100_ISPY2T1wce/dataset.json"
ISPY2_SCRIPTS_DIR="${ISPY2_DATASET_ROOT}/5_scripts_ispy2"
export PYTHONPATH="${ISPY2_SCRIPTS_DIR}:${PYTHONPATH:-}"

