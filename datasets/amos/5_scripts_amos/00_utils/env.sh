#!/usr/bin/env bash
# Source this file at the top of every amos pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#   source "$(dirname "$0")/00_utils/env.sh"       (from 5_scripts_amos root)
#
# AMOS is a full-pipeline dataset (CT + MRI, 15 abdominal organs). The training
# pipeline (01_create_splits → 04_train) is pending. Currently AMOS is used for
# INFERENCE ONLY: chaos-trained models are applied here as a cross-dataset probe.
# See datasets/amos/README.md and the "AMOS roadmap" there.
#
# ID convention: CT cases 0000–0499, MRI cases 0500–0599.

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROJECT_ROOT="$(cd "${DATASET_ROOT}/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/job_runner/run_job.sh"

export RAW_ROOT="${DATASET_ROOT}/0_raw_amos"
# AMOS ships as NIfTI (nnUNet-style) — raw layout lives under 0_raw_amos/amos22/.
# 1_BIDS_amos is populated by 00_00_bidsify.py when (if) needed.
export BIDS_ROOT="${DATASET_ROOT}/1_BIDS_amos/amos-abdominal"
export nnUNet_raw="${DATASET_ROOT}/2_nnUNet_amos/raw"
export nnUNet_preprocessed="${DATASET_ROOT}/2_nnUNet_amos/preprocessed"
# nnUNet_results: where nnUNet stores model checkpoints (native AMOS training, pending).
export nnUNet_results="${DATASET_ROOT}/8_results_amos/01_predictions/nnUNet"
export SPLITS_DIR="${DATASET_ROOT}/4_splits_amos"
export PREDICTIONS_ROOT="${DATASET_ROOT}/8_results_amos/01_predictions"
export METRICS_ROOT="${DATASET_ROOT}/8_results_amos/02_metrics"

export DATASET_NAME="amos"
# DATASET_ROLE: test-only — currently no AMOS-native models exist. We evaluate chaos-trained
# models here as a cross-dataset generalization probe (CT + MRI). This will change to "both"
# once the native AMOS training pipeline (01_create_splits → 04_train) is activated.
export DATASET_ROLE="test-only"
# nnUNet dataset ID for AMOS native training (reserved; populate when training is set up).
export NNUNET_DATASET_ID="Dataset070_AMOS"

export WANDB_PROJECT="mri_synthesis_seg_${DATASET_NAME}"

# ── Cross-dataset inference: chaos models ────────────────────────────────────
# Used by 05_predict/05_01_predict_chaos_common.sh until AMOS-native models exist.
export CHAOS_DATASET_ROOT="${DATASET_ROOT}/../chaos"
export CHAOS_PREDICTIONS_ROOT="${CHAOS_DATASET_ROOT}/8_results_chaos/01_predictions"
export CHAOS_NNUNET_RAW="${CHAOS_DATASET_ROOT}/2_nnUNet_chaos/raw"
export CHAOS_DATASET_ID="60"
export CHAOS_DS_NAME="Dataset060_CHAOS_MR_T1in"
export CHAOS_DATASET_JSON="${CHAOS_NNUNET_RAW}/${CHAOS_DS_NAME}/dataset.json"

SCRIPTS_DIR="${DATASET_ROOT}/5_scripts_amos"
CHAOS_SCRIPTS_DIR="${CHAOS_DATASET_ROOT}/5_scripts_chaos"
export PYTHONPATH="${SCRIPTS_DIR}:${CHAOS_SCRIPTS_DIR}:${PYTHONPATH:-}"
