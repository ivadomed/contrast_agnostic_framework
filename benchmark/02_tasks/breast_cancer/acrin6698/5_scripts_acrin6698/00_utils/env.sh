#!/usr/bin/env bash
# Source this at the top of every acrin6698 pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#
# acrin6698 = TCIA ACRIN-6698 (the I-SPY2 DWI sub-study), T0 baseline DWI arm --
# EVAL-ONLY cross-contrast test set for the I-SPY2-trained breast models (added
# 2026-09-30). Patients disjoint from the ispy2 training collection (0/385
# overlap, verified). Single test item `dwi` (highest-b trace DWI, b=800) with the
# trial's own "Whole Tumor Manual" DWI ROI -- a contrast neither training arm saw,
# so OOD for both. Bilateral axial FOV -> lesion-side unilateral crop (see
# 02_nnunet/02_01_convert_test.py). License: CC BY 4.0. Flat eval-only layout.

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET_NAME="acrin6698"
export MODEL_TYPE="acrin6698_model"
export DATASET_ROLE="eval_only"
export TRAINING_CONTRAST="${TRAINING_CONTRAST:-t1wce}"
export RUN_JOB_TIME_DEFAULT="${RUN_JOB_TIME_DEFAULT:-04:00:00}"

# (set BEFORE common_env.sh, which sources run_job and freezes its defaults)
# Small volumes (DWI ~256x256x40, cropped to half width) -- don't take the L40S defaults (16 CPU / 110G per GPU).
export RUN_JOB_CPUS_PER_GPU="${RUN_JOB_CPUS_PER_GPU:-8}"
export RUN_JOB_MEM_PER_GPU="${RUN_JOB_MEM_PER_GPU:-40G}"

export BIDS_SUBDIR="onc-breast-acrin6698"
CE_SUBDIRS="preprocessed splits"
source "${DATASET_ROOT}/../../../00_commun_scripts/00_00_utils/common_env.sh"

export METRICS_ROOT="${METRICS_ROOT:-${DATASET_ROOT}/8_results_acrin6698/02_metrics}"
export CHECKPOINTS_DIR="${DATASET_ROOT}/6_checkpoints_acrin6698"
export RESULTS_DIR="${DATASET_ROOT}/8_results_acrin6698"

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

