#!/usr/bin/env bash
# Train AugLab with the SynthSeg ("noEM") GPU augmentation config on CHAOS MR T1-in.
# Same AugLab trainer as synthseg_EM / auglab_default, but AUGLAB_PARAMS_GPU_JSON points
# at transform_params_gpu_default01-23_Synthseg.json (SynthSeg synthesis, NO EM).
# Uses Dataset060_CHAOS_MR_T1in (single T1 in-phase channel). WandB image panels logged
# automatically by the trainer (_log_wandb_images, axial).
#
# Placement: 4 folds on slots 0 and 1 (GPUs 0,1), 2 folds packed per slot/GPU:
#   folds 0,1 → slot 0 / GPU 0 ; folds 2,3 → slot 1 / GPU 1. Slots 2,3 left free.
#
# Usage:
#   bash 04_05_train_synthseg_noEM.sh                                         # auto: chaos_synthseg_noEM_train100_val000_<TS>
#   bash 04_05_train_synthseg_noEM.sh chaos_synthseg_noEM_train100_val000_<TS>  # explicit RUN_ID to resume

source "$(dirname "$0")/../00_utils/env.sh"

METHOD="synthseg_noEM_train100_val000"
TRAINER="nnUNetTrainerCHAOSAugLabDefault"
DATASET_ID="060"
DA_WORKERS=8
LOG_DIR="/tmp/nnunet_chaos_synthseg_noEM"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-500}"

# SynthSeg ("noEM") augmentation config (shared AugLab config — not dataset-specific).
AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_Synthseg.json"

# Save under 01_predictions/auglab (where auglab models live → predict/eval find them).
export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/auglab"

# 4 folds → slots 0 and 1 only (GPUs 0,1); 2 folds packed per slot/GPU.
export FOLD_SLOT_GPU="0,0,0 1,0,0 2,1,1 3,1,1"

source "$(dirname "$0")/04_00_common.sh" "$@"
