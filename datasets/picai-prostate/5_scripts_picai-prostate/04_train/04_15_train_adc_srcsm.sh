#!/usr/bin/env bash
# Train SRCSM (SemRandConv-3D, Thaler et al. 2025) on picai-prostate ADC (Dataset081_PICAI_ADC).
# csPCa lesion segmentation; 3 folds (0 1 2), 1 GPU/fold, 2000 epochs.
#
# Usage:
#   bash 04_15_train_adc_srcsm.sh                                  # auto RUN_ID
#   bash 04_15_train_adc_srcsm.sh picai-prostate_adc_srcsm_<TS>   # resume
source "$(dirname "$0")/../00_utils/env_adc.sh"

METHOD="srcsm"
TRAINER="nnUNetTrainerPICAIProstateAugLabDefault"
CATEGORY="auglab"
DATASET_ID="081"
DA_WORKERS=6
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_srcsm_semrandconv.json"
export AUGLAB_VAL_PARAMS_GPU_JSON=""

source "$(dirname "$0")/04_00_common.sh" "$@"
