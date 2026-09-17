#!/usr/bin/env bash
# Train SynthSeg (no EM) GPU augmentation on BraTS 2024 Glioma T2f/FLAIR.
# AugLab standard augs are deactivated (prob=0 in this config); only SynthSeg without EM.
# 3 folds (0 1 2), 1 GPU per fold, 2500 epochs.
#
# Usage:
#   bash 04_53_train_t2f_synthseg_noEM.sh           # auto RUN_ID
#   bash 04_53_train_t2f_synthseg_noEM.sh brats2024-glioma_t2f_synthseg_noEM_<TS>  # resume
export RUN_JOB_TIME_DEFAULT="2-23:00:00"  # 2500 epochs × ~60s/ep ≈ 42h
source "$(dirname "$0")/../00_utils/env_t2f.sh"

METHOD="synthseg_noEM"
TRAINER="nnUNetTrainerBraTS2024GliomaT2fAugLabDefault"
DATASET_ID="053"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_brats2024_t2f_synthseg_noEM"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2500}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_Synthseg.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
