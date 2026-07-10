#!/usr/bin/env bash
# AugLab augs + V26_6_2 GPU synthesis on BraTS 2024 Glioma T2w: 50% train synth / 0% val
# synth. Train pipeline: full AugLab augs + V26_6_2 synthesis at prob 0.5.
# Val pipeline:   clean (un-augmented), stock nnUNet validation_step.
# 3 folds (0 1 2), 1 GPU per fold, 2500 epochs.
#
# Usage:
#   bash 04_29_train_t2w_auglabAug_v26_6_2_train050_val000.sh           # auto RUN_ID
#   bash 04_29_train_t2w_auglabAug_v26_6_2_train050_val000.sh brats2024-glioma_t2w_auglabAug_v26_6_2_train050_val000_<TS>  # resume
source "$(dirname "$0")/../00_utils/env_t2w.sh"

METHOD="auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerBraTS2024GliomaT2wAugLabDefault"
DATASET_ID="052"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_brats2024_t2w_auglabAug_v26_6_2_train050_val000"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2500}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
