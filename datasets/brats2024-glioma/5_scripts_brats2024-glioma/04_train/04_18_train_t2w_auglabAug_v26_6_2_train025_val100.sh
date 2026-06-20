#!/usr/bin/env bash
# AugLab augs + V26_6_2 GPU synthesis on BraTS 2024 Glioma T2w: 25% train synth / 100% val synth.
# Train pipeline: full AugLab augs + V26_6_2 synthesis at prob 0.25.
# Val pipeline:   synth-only, V26_6_2 synthesis at prob 1.0 (VALsynthonly config).
# 4 folds, 1 GPU per fold, 2500 epochs.
#
# Usage:
#   bash 04_18_train_t2w_auglabAug_v26_6_2_train025_val100.sh           # auto RUN_ID
#   bash 04_18_train_t2w_auglabAug_v26_6_2_train025_val100.sh brats2024-glioma_t2w_auglabAug_v26_6_2_train025_val100_<TS>  # resume
source "$(dirname "$0")/../00_utils/env_t2w.sh"

METHOD="auglabAug_v26_6_2_train025_val100"
TRAINER="nnUNetTrainerBraTS2024GliomaT2wAugLabValSynth"
DATASET_ID="052"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_brats2024_t2w_auglabAug_v26_6_2_train025_val100"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2500}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train025.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
