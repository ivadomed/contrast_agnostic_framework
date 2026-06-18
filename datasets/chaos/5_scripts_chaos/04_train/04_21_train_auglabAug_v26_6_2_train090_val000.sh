#!/usr/bin/env bash
# AugLab default augs + V26_6_2 GPU transform @90% train, val synth 0% (clean validation).
# The AugLabV26_6_2 trainer's validation_uses_augmentation=True only changes the WandB
# viz panel — actual validation runs on clean data (stock nnUNet validation_step).
# All 4 folds, 1 fold/GPU, 200 epochs.
#
# Usage:
#   bash 04_21_train_auglabAug_v26_6_2_train090_val000.sh                                      # auto: chaos_auglabAug_v26_6_2_train090_val000_<TS>
#   bash 04_21_train_auglabAug_v26_6_2_train090_val000.sh chaos_auglabAug_v26_6_2_train090_val000_<TS>  # resume
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="auglabAug_v26_6_2_train090_val000"
TRAINER="nnUNetTrainerCHAOSAugLabV26_6_2"
DATASET_ID="060"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_chaos_auglabAug_v26_6_2_train090_val000"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-200}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train090.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
