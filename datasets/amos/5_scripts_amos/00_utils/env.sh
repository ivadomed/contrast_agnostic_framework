#!/usr/bin/env bash
# Source this file at the top of every amos pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#   source "$(dirname "$0")/00_utils/env.sh"       (from 5_scripts_amos root)
#
# AMOS is a full-pipeline dataset (CT + MRI, 15 abdominal organs). The training
# pipeline (01_create_splits → 04_train) is pending. Currently AMOS is used for
# INFERENCE ONLY: chaos-trained models are applied here as a cross-dataset probe.
# See datasets/amos/README.md and the "AMOS roadmap" there.
# ID convention: CT cases 0000–0499, MRI cases 0500–0599.
#
# This is the amos "config": it sets the dataset-specific values, then sources
# datasets/00_commun_scripts/00_00_utils/common_env.sh for the standard paths.

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET_NAME="amos"
# DATASET_ROLE: test-only — no AMOS-native models exist yet. We evaluate chaos-trained
# models here as a cross-dataset generalization probe (CT + MRI). Becomes "both" once
# the native AMOS training pipeline (01_create_splits → 04_train) is activated.
DATASET_ROLE="test-only"
# nnUNet dataset ID for AMOS native training (reserved; populate when training is set up).
export NNUNET_DATASET_ID="Dataset070_AMOS"

# ── Cross-dataset inference: chaos models ────────────────────────────────────
# Used by 05_predict/05_01_predict_chaos_common.sh until AMOS-native models exist.
# Set BEFORE common_env so CE_EXTRA_PYTHONPATH can reference CHAOS_SCRIPTS_DIR. The
# ${VAR:-default} forms let amos's env_t2spir.sh pre-export the T2spir chaos vars.
export CHAOS_DATASET_ROOT="${DATASET_ROOT}/../chaos"
export CHAOS_PREDICTIONS_ROOT="${CHAOS_DATASET_ROOT}/8_results_chaos/01_predictions"
export CHAOS_NNUNET_RAW="${CHAOS_DATASET_ROOT}/2_nnUNet_chaos/raw"
export CHAOS_DATASET_ID="${CHAOS_DATASET_ID:-60}"
export CHAOS_DS_NAME="${CHAOS_DS_NAME:-Dataset060_CHAOS_MR_T1in}"
export CHAOS_DATASET_JSON="${CHAOS_NNUNET_RAW}/${CHAOS_DS_NAME}/dataset.json"
export CHAOS_TRAINING_CONTRAST="${CHAOS_TRAINING_CONTRAST:-t1in}"
export CHAOS_MODEL_TYPE="chaos_model"
CHAOS_SCRIPTS_DIR="${CHAOS_DATASET_ROOT}/5_scripts_chaos"

# common_env config (plain vars — consumed by common_env, not exported to the env):
BIDS_SUBDIR="amos-abdominal"             # → BIDS_ROOT under 1_BIDS_<name>/
CE_SUBDIRS="raw preprocessed splits"      # 0_raw + 2_nnUNet/preprocessed + 4_splits
CE_EXTRA_PYTHONPATH="${CHAOS_SCRIPTS_DIR}"   # add chaos scripts so chaos trainer classes resolve
source "${DATASET_ROOT}/../00_commun_scripts/00_00_utils/common_env.sh"

# nnUNet_results: native AMOS training (pending) would write checkpoints here.
export nnUNet_results="${DATASET_ROOT}/8_results_amos/01_predictions/nnUNet"
