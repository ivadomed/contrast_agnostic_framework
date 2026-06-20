#!/usr/bin/env bash
# Source this file at the top of every brats2024-glioma pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#   source "$(dirname "$0")/00_utils/env.sh"       (from 5_scripts_brats2024-glioma root)

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROJECT_ROOT="$(cd "${DATASET_ROOT}/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/job_runner/run_job.sh"

export BIDS_ROOT="${DATASET_ROOT}/1_BIDS_brats2024-glioma/glioma-brain-brats2024"
export nnUNet_raw="${DATASET_ROOT}/2_nnUNet_brats2024-glioma/raw"
export nnUNet_preprocessed="${DATASET_ROOT}/2_nnUNet_brats2024-glioma/preprocessed"
# nnUNet_results: where nnUNet stores model checkpoints for nnUNet-family methods.
# Path includes model_type and training_contrast so each training context is isolated.
# Auglab wrappers override NNUNET_RESULTS_BASE to 01_predictions/brats2024_glioma_model/<contrast>/auglab.
# Conditional assignment: env_t2w.sh (and similar contrast overrides) sets this BEFORE sourcing
# 04_00_common.sh, which re-sources this file — the ${VAR:-default} form preserves the override.
export nnUNet_results="${nnUNet_results:-${DATASET_ROOT}/8_results_brats2024-glioma/01_predictions/brats2024_glioma_model/t1n/nnUNet}"
export SPLITS_DIR="${DATASET_ROOT}/4_splits_brats2024-glioma"
# Prediction output root: 01_predictions/{model_type}/{training_contrast}/{category}/{run_id}/fold{k}/{contrast}/
export PREDICTIONS_ROOT="${DATASET_ROOT}/8_results_brats2024-glioma/01_predictions"
# Metrics output root: 02_metrics/{model_type}/{training_contrast}/{category}_{run_id}/fold{k}/
export METRICS_ROOT="${DATASET_ROOT}/8_results_brats2024-glioma/02_metrics"

export DATASET_NAME="brats2024-glioma"
# DATASET_ROLE: training | test-only | both
# "test-only" datasets have no 04_train/ or 05_predict/ steps; they evaluate
# foreign models on their own test set. "both" datasets do all steps AND receive
# cross-dataset predictions (named {source_dataset}_*) in 01_predictions/.
export DATASET_ROLE="training"
export NNUNET_DATASET_ID="Dataset050_BraTS2024Glioma"
export WANDB_PROJECT="mri_synthesis_seg_${DATASET_NAME}"

# Training contrast: the MRI contrast used to train models in this dataset.
# Used in run names ({dataset}_{training_contrast}_{method}_{ts}) and result paths.
# Conditional assignment: env_t2w.sh sets this before 04_00_common.sh re-sources this file.
export TRAINING_CONTRAST="${TRAINING_CONTRAST:-t1n}"
# Model type label: identifies the source dataset for predictions (own or cross-dataset).
export MODEL_TYPE="brats2024_glioma_model"

# Make the brats2024_glioma Python package importable (trainers, etc.)
SCRIPTS_DIR="${DATASET_ROOT}/5_scripts_brats2024-glioma"
export PYTHONPATH="${SCRIPTS_DIR}:${PYTHONPATH:-}"
