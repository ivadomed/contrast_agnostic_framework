#!/usr/bin/env bash
# Source this file at the top of every chaos pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#   source "$(dirname "$0")/00_utils/env.sh"       (from 5_scripts_chaos root)

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROJECT_ROOT="$(cd "${DATASET_ROOT}/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/job_runner/run_job.sh"

export RAW_ROOT="${DATASET_ROOT}/0_raw_chaos"
export BIDS_ROOT="${DATASET_ROOT}/1_BIDS_chaos/chaos-abdominal"
export nnUNet_raw="${DATASET_ROOT}/2_nnUNet_chaos/raw"
export nnUNet_preprocessed="${DATASET_ROOT}/2_nnUNet_chaos/preprocessed"
# nnUNet_results: where nnUNet stores model checkpoints for nnUNet-family methods.
# Path includes model_type and training_contrast so each training context is isolated.
# Conditional assignment: env_t2spir.sh (and any future contrast env) sets this before
# sourcing env.sh; 04_00_common.sh re-sources env.sh so we must not clobber it.
export nnUNet_results="${nnUNet_results:-${DATASET_ROOT}/8_results_chaos/01_predictions/chaos_model/t1in/nnUNet}"
export SPLITS_DIR="${DATASET_ROOT}/4_splits_chaos"
# Prediction output root: 01_predictions/{model_type}/{training_contrast}/{category}/{run_id}/fold{k}/{modality}/
export PREDICTIONS_ROOT="${DATASET_ROOT}/8_results_chaos/01_predictions"
# Metrics output root: 02_metrics/{model_type}/{training_contrast}/{category}_{run_id}/fold{k}/
export METRICS_ROOT="${DATASET_ROOT}/8_results_chaos/02_metrics"

export DATASET_NAME="chaos"
# DATASET_ROLE: training | test-only | both
# "test-only" datasets have no 04_train/ or 05_predict/ steps; they evaluate
# foreign models on their own test set. "both" datasets do all steps AND receive
# cross-dataset predictions (named {source_dataset}_*) in 01_predictions/.
export DATASET_ROLE="training"
# Train dataset = MR T1-DUAL in-phase only (single channel). See 02_nnunet/02_00_convert.py.
export NNUNET_DATASET_ID="Dataset060_CHAOS_MR_T1in"

# Training contrast: the MRI contrast used to train models in this dataset.
# Used in run names ({dataset}_{training_contrast}_{method}_{ts}) and result paths.
# Conditional assignment: env_t2spir.sh sets this before 04_00_common.sh re-sources env.sh.
export TRAINING_CONTRAST="${TRAINING_CONTRAST:-t1in}"
# Model type label: identifies the source dataset for predictions (own or cross-dataset).
export MODEL_TYPE="chaos_model"

# Dataset-specific WandB project (keep per-dataset runs isolated; same convention for future datasets).
export WANDB_PROJECT="mri_synthesis_seg_${DATASET_NAME}"

# Make a future chaos Python package importable (trainers, etc.)
SCRIPTS_DIR="${DATASET_ROOT}/5_scripts_chaos"
export PYTHONPATH="${SCRIPTS_DIR}:${PYTHONPATH:-}"
