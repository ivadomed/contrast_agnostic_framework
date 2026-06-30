#!/usr/bin/env bash
# Source this file at the top of every on-harmony pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#   source "$(dirname "$0")/00_utils/env.sh"       (from 5_scripts_on-harmony root)
#
# This is the on-harmony "config": it sets the dataset-specific values, then sources
# datasets/00_commun_scripts/00_00_utils/common_env.sh for the standard paths.
#
# STANDARD results layout (identical to chaos, since 2026-06-28): trained models are
# CO-LOCATED with their predictions under
#   01_predictions/<MODEL_TYPE>/<TRAINING_CONTRAST>/<nnUNet|auglab>/<RUN_ID>/
# and metrics under 02_metrics/<MODEL_TYPE>/<TRAINING_CONTRAST>/. nnUNet_results below
# points at the nnUNet-category base (the default for nnUNet-family methods); auglab
# per-method train scripts override NNUNET_RESULTS_BASE → .../<contrast>/auglab (exactly
# as chaos does). The legacy 02_nnUNet_results/runs + 01_results trees were archived to
# <project_root>/_archive/on-harmony_legacy_results_20260628/.

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET_NAME="on-harmony"
# DATASET_ROLE training|test-only|both — defaults to "training" in common_env.
export NNUNET_DATASET_ID="Dataset030_OnHarmonyT1w"
export MODEL_TYPE="on_harmony_model"
export TRAINING_CONTRAST="${TRAINING_CONTRAST:-T1w}"
# Conditional so a contrast env (e.g. env_t2w.sh) that pre-exports this survives the
# re-source done by 04_00_common.sh. 96h matches the time limit T1w training has run
# with since the set_slot → Slurm migration (2026-06-19).
export RUN_JOB_TIME_DEFAULT="${RUN_JOB_TIME_DEFAULT:-96:00:00}"

# common_env config (plain vars — consumed by common_env, not exported to the env):
CE_SUBDIRS="preprocessed splits"          # has 2_nnUNet/preprocessed + 4_splits (no 0_raw)
source "${DATASET_ROOT}/../00_commun_scripts/00_00_utils/common_env.sh"

# 1_BIDS_on-harmony has no leaf subdir (unlike chaos-abdominal etc.), so set BIDS_ROOT
# explicitly rather than via common_env's BIDS_SUBDIR mechanism.
export BIDS_ROOT="${DATASET_ROOT}/1_BIDS_on-harmony"
# Standard layout (see header). nnUNet_results = the nnUNet-category model base for the
# active training contrast; nnU-Net writes <RUN_ID>/<trainer>/fold_N under it (the shared
# train driver appends RUN_ID). Mirrors chaos's .../01_predictions/chaos_model/t1in/nnUNet.
export nnUNet_results="${DATASET_ROOT}/8_results_on-harmony/01_predictions/${MODEL_TYPE}/${TRAINING_CONTRAST}/nnUNet"
export CHECKPOINTS_DIR="${DATASET_ROOT}/6_checkpoints_on-harmony"
export RESULTS_DIR="${DATASET_ROOT}/8_results_on-harmony"
