#!/usr/bin/env bash
# Source this file at the top of every brats2017 pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#   source "$(dirname "$0")/00_utils/env.sh"       (from 5_scripts_brats2017 root)

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export RAW_ROOT="${DATASET_ROOT}/0_raw_brats2017"
export nnUNet_raw="${DATASET_ROOT}/2_nnUNet_brats2017/raw"
export nnUNet_preprocessed="${DATASET_ROOT}/2_nnUNet_brats2017/preprocessed"
export nnUNet_results="${DATASET_ROOT}/8_results_brats2017/02_nnUNet_results"
export SPLITS_DIR="${DATASET_ROOT}/4_splits_brats2017"
export CHECKPOINTS_DIR="${DATASET_ROOT}/6_checkpoints_brats2017"
export RESULTS_DIR="${DATASET_ROOT}/8_results_brats2017"

export DATASET_NAME="brats2017"
# BraTS uses four nnUNet dataset IDs (one per contrast)
export NNUNET_DATASET_ID_FLAIR="Dataset021_BraTSFlair_gen_raw"
export NNUNET_DATASET_ID_T1W="Dataset022_BraTST1w_gen_raw"
export NNUNET_DATASET_ID_T2W="Dataset023_BraTST2w_gen_raw"
export NNUNET_DATASET_ID_T1GD="Dataset024_BraTST1gd_gen_raw"
