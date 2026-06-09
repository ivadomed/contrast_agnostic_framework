#!/usr/bin/env bash
# Source this file at the top of every brats2024-glioma pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#   source "$(dirname "$0")/00_utils/env.sh"       (from 5_scripts_brats2024-glioma root)

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export BIDS_ROOT="${DATASET_ROOT}/1_BIDS_brats2024-glioma/glioma-brain-brats2024"
export nnUNet_raw="${DATASET_ROOT}/2_nnUNet_brats2024-glioma/raw"
export nnUNet_preprocessed="${DATASET_ROOT}/2_nnUNet_brats2024-glioma/preprocessed"
export nnUNet_results="${DATASET_ROOT}/8_results_brats2024-glioma/02_nnUNet_results"
export SPLITS_DIR="${DATASET_ROOT}/4_splits_brats2024-glioma"
export CHECKPOINTS_DIR="${DATASET_ROOT}/6_checkpoints_brats2024-glioma"
export RESULTS_DIR="${DATASET_ROOT}/8_results_brats2024-glioma"

export DATASET_NAME="brats2024-glioma"
export NNUNET_DATASET_ID="Dataset050_BraTS2024Glioma"

# Make the brats2024_glioma Python package importable (trainers, etc.)
SCRIPTS_DIR="${DATASET_ROOT}/5_scripts_brats2024-glioma"
export PYTHONPATH="${SCRIPTS_DIR}:${PYTHONPATH:-}"
