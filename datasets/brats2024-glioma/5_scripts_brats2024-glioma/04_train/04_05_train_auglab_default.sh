#!/usr/bin/env bash
# Train AugLab default GPU augmentation baseline on BraTS 2024 Glioma (T1n).
# Uses Dataset051_BraTS2024GliomaT1n (single T1n channel).
# Results saved under 8_results_brats2024-glioma/01_results/auglab/
# Usage: bash 04_05_train_auglab_default.sh [RUN_ID]

source "$(dirname "$0")/../00_utils/env.sh"

METHOD="auglab_default"
TRAINER="nnUNetTrainerBraTS2024GliomaAugLabDefault"
DATASET_ID="051"
DA_WORKERS=8
LOG_DIR="/tmp/nnunet_brats2024_auglab_default"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-1000}"

# Point to the AugLab default augmentation config (no synthesis — standard aug only)
AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23.json"

# Save under 01_results/auglab (not 02_nnUNet_results); NNUNET_RESULTS_BASE is read
# by 04_00_common.sh after it re-sources env.sh, so this override survives.
DATASET_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export NNUNET_RESULTS_BASE="${DATASET_ROOT}/8_results_brats2024-glioma/01_results/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
