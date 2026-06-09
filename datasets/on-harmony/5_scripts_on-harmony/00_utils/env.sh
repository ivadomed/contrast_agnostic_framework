#!/usr/bin/env bash
# Source this file at the top of every on-harmony pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#   source "$(dirname "$0")/00_utils/env.sh"       (from 5_scripts_on-harmony root)

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export BIDS_ROOT="${DATASET_ROOT}/1_BIDS_on-harmony"
export nnUNet_raw="${DATASET_ROOT}/2_nnUNet_on-harmony/raw"
export nnUNet_preprocessed="${DATASET_ROOT}/2_nnUNet_on-harmony/preprocessed"
export nnUNet_results="${DATASET_ROOT}/8_results_on-harmony/02_nnUNet_results"
export SPLITS_DIR="${DATASET_ROOT}/4_splits_on-harmony"
export CHECKPOINTS_DIR="${DATASET_ROOT}/6_checkpoints_on-harmony"
export RESULTS_DIR="${DATASET_ROOT}/8_results_on-harmony"

export DATASET_NAME="on-harmony"
export NNUNET_DATASET_ID="Dataset030_OnHarmonyT1w"

# Make the on_harmony Python package importable (trainers, dataloader).
# Any script that launches nnUNet training must source this file first.
SCRIPTS_DIR="${DATASET_ROOT}/5_scripts_on-harmony"
export PYTHONPATH="${SCRIPTS_DIR}:${PYTHONPATH:-}"
