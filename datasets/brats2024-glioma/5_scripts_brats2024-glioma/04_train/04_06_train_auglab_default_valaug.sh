#!/usr/bin/env bash
# Train AugLab default config on BraTS 2024 Glioma (T1n) WITH validation augmented at 100%.
# Same default paper config as 04_05, but the GPU transform pipeline is also applied
# to 100% of validation batches (trainer: ...AugLabDefaultValAug) so the val
# distribution matches training under domain randomization.
# Uses Dataset051_BraTS2024GliomaT1n (single T1n channel).
# Results saved under 8_results_brats2024-glioma/01_results/auglab_valaug/
# Usage: bash 04_06_train_auglab_default_valaug.sh [RUN_ID]

source "$(dirname "$0")/../00_utils/env.sh"

METHOD="auglab_default_valaug"
TRAINER="nnUNetTrainerBraTS2024GliomaAugLabDefaultValAug"
DATASET_ID="051"
DA_WORKERS=8
LOG_DIR="/tmp/nnunet_brats2024_auglab_default_valaug"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-1000}"

# Same AugLab default augmentation config as the paper-faithful run (04_05).
AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23.json"

# Save under 01_results/auglab_valaug (separate from the paper-faithful auglab run);
# NNUNET_RESULTS_BASE is read by 04_00_common.sh after it re-sources env.sh.
DATASET_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab_valaug"

source "$(dirname "$0")/04_00_common.sh" "$@"
