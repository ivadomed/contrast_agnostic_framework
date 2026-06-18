#!/usr/bin/env bash
# AugLab default augmentations + V26_6_2 GPU transform @50% (train), val synth 0%.
# Train synth = config prob 0.5; validation runs on clean data (stock nnUNet
# validation_step). The AugLabV26_6_2 trainer's validation_uses_augmentation=True
# only changes the WandB *viz panel* — actual validation is un-augmented (val 0%).
# Runs on set_slot 3 / GPU 3.
#
# Usage:
#   bash 04_09_train_auglabAug_v26_6_2_train050_val000.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerCHAOSAugLabV26_6_2"
DATASET_ID="060"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_chaos_auglabAug_v26_6_2_train050_val000"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-300}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/auglab"

export SINGLE_FOLD="${SINGLE_FOLD:-0}"
export SINGLE_SLOT="${SINGLE_SLOT:-3}"
export SINGLE_GPU="${SINGLE_GPU:-3}"

source "$(dirname "$0")/04_00_common.sh" "$@"
