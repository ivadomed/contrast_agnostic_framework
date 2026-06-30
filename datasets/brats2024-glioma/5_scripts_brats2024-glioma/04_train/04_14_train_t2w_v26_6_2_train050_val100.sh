#!/usr/bin/env bash
# Train V26_6_2 on BraTS 2024 Glioma T2w: 50% train synth / 100% val synth.
# GPU synthesis via nnUNetTrainerBraTS2024GliomaT2wV26_6_2_train050_val100.
# 4 folds, 1 GPU per fold, 2500 epochs.
#
# Usage:
#   bash 04_14_train_t2w_v26_6_2_train050_val100.sh           # auto RUN_ID
#   bash 04_14_train_t2w_v26_6_2_train050_val100.sh brats2024-glioma_t2w_v26_6_2_train050_val100_<TS>  # resume
source "$(dirname "$0")/../00_utils/env_t2w.sh"

METHOD="v26_6_2_train050_val100"
TRAINER="nnUNetTrainerBraTS2024GliomaT2wV26_6_2_train050_val100"
DATASET_ID="052"
DA_WORKERS=16
LOG_DIR="/tmp/nnunet_brats2024_t2w_v26_6_2_train050_val100"
# torch.compile requires Triton which is not available on Vulcan — keep disabled.
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2500}"

# v26_6_2 now runs the AugLab GPU contrast transform (synth + standard spatial DA,
# no other AugLab augs). Both configs required by nnUNetTrainerBraTS2024GliomaV26_6_2*.
_AUGLAB_CONFIGS="$(cd "${PROJECT_ROOT}/sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${_AUGLAB_CONFIGS}/transform_params_gpu_v26_6_2_synth_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${_AUGLAB_CONFIGS}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"
source "$(dirname "$0")/04_00_common.sh" "$@"
