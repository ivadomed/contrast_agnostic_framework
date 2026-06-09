#!/usr/bin/env bash
# Source this file at the top of every spider-spine pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#   source "$(dirname "$0")/00_utils/env.sh"       (from 5_scripts_spider-spine root)

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export RAW_ROOT="${DATASET_ROOT}/0_raw_spider-spine"
export nnUNet_raw="${DATASET_ROOT}/2_nnUNet_spider-spine/raw"
export nnUNet_preprocessed="${DATASET_ROOT}/2_nnUNet_spider-spine/preprocessed"
export nnUNet_results="${DATASET_ROOT}/8_results_spider-spine/02_nnUNet_results"
export SPLITS_DIR="${DATASET_ROOT}/4_splits_spider-spine"
export CHECKPOINTS_DIR="${DATASET_ROOT}/6_checkpoints_spider-spine"
export RESULTS_DIR="${DATASET_ROOT}/8_results_spider-spine"

export DATASET_NAME="spider-spine"
export NNUNET_DATASET_ID_T1="Dataset122_SpiderSpine_t1"
export NNUNET_DATASET_ID_T2="Dataset123_SpiderSpine_t2"
export NNUNET_DATASET_ID_T2SPACE="Dataset124_SpiderSpine_t2space"
