#!/usr/bin/env bash
# Train SRCSM (SemRandConv-3D, Thaler et al. 2025) on ON-Harmony T1w, 2000 epochs.
# AugLab-category method: reuses the AugLab-default trainer, selected purely by config.
# Usage: bash 04_20_train_t1w_srcsm.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="srcsm"
TRAINER="nnUNetTrainerOnHarmonyAugLabDefault"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_on-harmony_t1w_srcsm"
export NNUNET_NUM_EPOCHS=2000

# Must export explicitly: the shared train driver exports AUGLAB_PARAMS_GPU_JSON as an
# empty string when unset, which defeats the trainer's "unset → default config" fallback
# (empty ≠ unset) → open('') crash. Matches the sibling auglab scripts (04_15/16/17/18).
_AUGLAB_CONFIGS="$(cd "${PROJECT_ROOT}/sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${_AUGLAB_CONFIGS}/transform_params_gpu_srcsm_semrandconv.json"

# Save under 01_predictions/<contrast>/auglab (auglab-category models → predict/eval find them).
export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
