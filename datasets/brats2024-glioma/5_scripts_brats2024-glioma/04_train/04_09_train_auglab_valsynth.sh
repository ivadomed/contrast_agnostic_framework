#!/usr/bin/env bash
# AugLab default augs + V26_6_2 GPU transform @50% (train), and REAL synth-only
# validation @100% (val). BraTS 2024 Glioma (Dataset051, single T1n channel).
#
# Train pipeline: full AugLab augs + V26 synth at prob 0.5 (train050 config).
# Val pipeline:   synth-only, V26 synth at prob 1.0 (VALsynthonly config) — applied
#                 in a custom validation_step (real val synth, affects metrics).
#
# Usage:
#   bash 04_09_train_auglab_valsynth.sh [RUN_ID]

source "$(dirname "$0")/../00_utils/env.sh"

METHOD="auglab_valsynth_train050_val100"
TRAINER="nnUNetTrainerBraTS2024GliomaAugLabValSynth"
DATASET_ID="051"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_brats2024_auglab_valsynth"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-1000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"

DATASET_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export NNUNET_RESULTS_BASE="${DATASET_ROOT}/8_results_brats2024-glioma/01_results/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
