#!/usr/bin/env bash
# Train SynthSeg+EM GPU augmentation on BraTS 2024 Glioma T1n.
# AugLab standard augs are deactivated (prob=0 in this config); only SynthSeg+EM synthesis.
# 4 folds, 1 GPU per fold, 2500 epochs.
#
# Usage:
#   bash 04_22_train_t1n_synthseg_EM.sh           # auto RUN_ID
#   bash 04_22_train_t1n_synthseg_EM.sh brats2024-glioma_t1n_synthseg_EM_<TS>  # resume
export RUN_JOB_TIME_DEFAULT="2-23:00:00"  # 2500 epochs × ~60s/ep ≈ 42h
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="synthseg_EM"
TRAINER="nnUNetTrainerBraTS2024GliomaAugLabDefault"
DATASET_ID="051"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_brats2024_t1n_synthseg_EM"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2500}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_Synthseg_EM.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
