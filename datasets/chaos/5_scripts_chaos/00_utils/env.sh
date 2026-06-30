#!/usr/bin/env bash
# Source this file at the top of every chaos pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#   source "$(dirname "$0")/00_utils/env.sh"       (from 5_scripts_chaos root)
#
# This is the chaos "config": it sets the dataset-specific values, then sources
# datasets/00_commun_scripts/00_00_utils/common_env.sh, which derives every standard
# 9-subdir path from $DATASET_NAME. See that file for the full contract.

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET_NAME="chaos"
# DATASET_ROLE training|test-only|both — defaults to "training" in common_env.
# Train dataset = MR T1-DUAL in-phase only (single channel). See 02_nnunet/02_00_convert.py.
export NNUNET_DATASET_ID="Dataset060_CHAOS_MR_T1in"
# Model type label: identifies the source dataset for predictions (own or cross-dataset).
export MODEL_TYPE="chaos_model"
# Training contrast used to train models here. Conditional so env_t2spir.sh's
# pre-export survives the re-source done by 04_00_common.sh.
export TRAINING_CONTRAST="${TRAINING_CONTRAST:-t1in}"

# common_env config (plain vars — consumed by common_env, not exported to the env):
BIDS_SUBDIR="chaos-abdominal"            # → BIDS_ROOT under 1_BIDS_<name>/
CE_SUBDIRS="raw preprocessed splits"      # this dataset has 0_raw + 2_nnUNet/preprocessed + 4_splits
source "${DATASET_ROOT}/../00_commun_scripts/00_00_utils/common_env.sh"

# nnUNet_results: where nnUNet stores model checkpoints for nnUNet-family methods.
# Path includes model_type and training_contrast so each training context is isolated.
# Conditional so a contrast env that pre-exports it (e.g. before a re-source) wins;
# chaos's env_t2spir.sh instead sources this file first and then overrides unconditionally.
export nnUNet_results="${nnUNet_results:-${DATASET_ROOT}/8_results_chaos/01_predictions/chaos_model/t1in/nnUNet}"
