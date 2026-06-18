#!/usr/bin/env bash
# AugLab default augs + V26_6_2 GPU transform @25% (train), synth-only val @100%.
# Usage: bash 04_11_train_auglabAug_v26_6_2_train025_val100.sh [RUN_ID]

source "$(dirname "$0")/../00_utils/env.sh"

METHOD="auglabAug_v26_6_2_train025_val100"
TRAINER="nnUNetTrainerBraTS2024GliomaAugLabValSynth"
DATASET_ID="051"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_brats2024_auglabAug_v26_6_2_train025_val100"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-1500}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train025.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"

DATASET_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export NNUNET_RESULTS_BASE="${DATASET_ROOT}/8_results_brats2024-glioma/01_results/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
