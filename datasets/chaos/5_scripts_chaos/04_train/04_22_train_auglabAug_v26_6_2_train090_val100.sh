#!/usr/bin/env bash
# AugLab default augs + V26_6_2 GPU transform @90% train, 100% val synth.
# Train pipeline: full AugLab augs + V26 synth at prob 0.9.
# Val pipeline:   synth-only, V26 synth at prob 1.0 (VALsynthonly config).
# All 4 folds, 1 fold/GPU, 200 epochs.
#
# Usage:
#   bash 04_22_train_auglabAug_v26_6_2_train090_val100.sh                                      # auto: chaos_auglabAug_v26_6_2_train090_val100_<TS>
#   bash 04_22_train_auglabAug_v26_6_2_train090_val100.sh chaos_auglabAug_v26_6_2_train090_val100_<TS>  # resume
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="auglabAug_v26_6_2_train090_val100"
TRAINER="nnUNetTrainerCHAOSAugLabValSynth"
DATASET_ID="060"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_chaos_auglabAug_v26_6_2_train090_val100"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-200}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train090.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
