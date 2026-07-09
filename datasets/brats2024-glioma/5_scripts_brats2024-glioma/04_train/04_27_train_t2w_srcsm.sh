#!/usr/bin/env bash
# Train SRCSM (SemRandConv-3D, Thaler et al. 2025) augmentation on BraTS 2024 Glioma T2w.
# 7th comparison arm (additional to the 6 usual methods); reuses the AugLab-default trainer.
# 4 folds, 1 GPU per fold, 2500 epochs.
#
# Usage:
#   bash 04_27_train_t2w_srcsm.sh           # auto RUN_ID
#   bash 04_27_train_t2w_srcsm.sh brats2024-glioma_t2w_srcsm_<TS>  # resume
source "$(dirname "$0")/../00_utils/env_t2w.sh"

METHOD="srcsm"
TRAINER="nnUNetTrainerBraTS2024GliomaT2wAugLabDefault"
DATASET_ID="052"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_brats2024_t2w_srcsm"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2500}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_srcsm_semrandconv.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
