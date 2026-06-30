#!/usr/bin/env bash
# Source this file at the top of every brats2024-glioma pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#   source "$(dirname "$0")/00_utils/env.sh"       (from 5_scripts_brats2024-glioma root)
#
# This is the brats2024-glioma "config": it sets the dataset-specific values, then
# sources datasets/00_commun_scripts/00_00_utils/common_env.sh, which derives every
# standard 9-subdir path from $DATASET_NAME. See that file for the full contract.

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET_NAME="brats2024-glioma"
# DATASET_ROLE training|test-only|both — defaults to "training" in common_env.
export NNUNET_DATASET_ID="Dataset050_BraTS2024Glioma"
# Model type label: identifies the source dataset for predictions (own or cross-dataset).
export MODEL_TYPE="brats2024_glioma_model"
# Training contrast: the MRI contrast used to train models in this dataset. Used in run
# names ({dataset}_{training_contrast}_{method}_{ts}) and result paths. Conditional so
# env_t2w.sh can pre-export it before this file (re-)sources.
export TRAINING_CONTRAST="${TRAINING_CONTRAST:-t1n}"

# common_env config (plain vars — consumed by common_env, not exported to the env):
BIDS_SUBDIR="glioma-brain-brats2024"     # → BIDS_ROOT under 1_BIDS_<name>/
CE_SUBDIRS="preprocessed splits"          # this dataset has 2_nnUNet/preprocessed + 4_splits (no 0_raw)
source "${DATASET_ROOT}/../00_commun_scripts/00_00_utils/common_env.sh"

# nnUNet_results: where nnUNet stores model checkpoints for nnUNet-family methods.
# Path includes model_type and training_contrast so each training context is isolated.
# Auglab wrappers override NNUNET_RESULTS_BASE to .../<contrast>/auglab. Conditional so
# env_t2w.sh's pre-export (set before this file is re-sourced by 04_00_common.sh) wins.
export nnUNet_results="${nnUNet_results:-${DATASET_ROOT}/8_results_brats2024-glioma/01_predictions/brats2024_glioma_model/t1n/nnUNet}"
