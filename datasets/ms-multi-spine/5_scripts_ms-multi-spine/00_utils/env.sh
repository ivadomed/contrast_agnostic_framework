#!/usr/bin/env bash
# Source this file at the top of every ms-multi-spine pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#   source "$(dirname "$0")/00_utils/env.sh"       (from 5_scripts_ms-multi-spine root)

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export BIDS_ROOT="${DATASET_ROOT}/1_BIDS_ms-multi-spine"
export nnUNet_raw="${DATASET_ROOT}/2_nnUNet_ms-multi-spine/raw"
export nnUNet_preprocessed="${DATASET_ROOT}/2_nnUNet_ms-multi-spine/preprocessed"
export nnUNet_results="${DATASET_ROOT}/8_results_ms-multi-spine/02_nnUNet_results"
export SPLITS_DIR="${DATASET_ROOT}/4_splits_ms-multi-spine"
export CHECKPOINTS_DIR="${DATASET_ROOT}/6_checkpoints_ms-multi-spine"
export RESULTS_DIR="${DATASET_ROOT}/8_results_ms-multi-spine"

export DATASET_NAME="ms-multi-spine"
export NNUNET_DATASET_ID="Dataset104_MSMultiSpine"
