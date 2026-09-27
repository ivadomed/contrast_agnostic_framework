#!/usr/bin/env bash
# Train V26_6_2 (train_synth_prob=0.50, val_synth_prob=1.0) on ON-Harmony dwi_ap, 2000 epochs.
# Usage: bash 04_42_train_dwi_ap_v26_6_2_train050_val100.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env_dwi.sh"
METHOD="v26_6_2_train050_val100"
TRAINER="nnUNetTrainerOnHarmonyV26_6_2_train050_val100"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_on-harmony_dwi_ap_v26_6_2_train050_val100"
export DATASET_ID="033"
export NNUNET_NUM_EPOCHS=2000

# v26_6_2 now runs the AugLab GPU contrast transform (synth + standard spatial DA, no
# other AugLab augs). Both configs are required by nnUNetTrainerOnHarmonyAugLabValSynth.
_AUGLAB_CONFIGS="$(cd "${PROJECT_ROOT}/sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${_AUGLAB_CONFIGS}/transform_params_gpu_v26_6_2_synth_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${_AUGLAB_CONFIGS}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"

# nnUNet-category run: set the results base EXPLICITLY. 04_00_common.sh re-sources env.sh, which
# unconditionally re-exports nnUNet_results to the repo path and would clobber a cluster override
# (on TamIA that lands checkpoints in /project — 498K/500K files — instead of $SCRATCH).
# PREDICTIONS_ROOT is set conditionally in common_env, so it survives; on Vulcan this resolves to
# the identical path env.sh would have used.
export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/nnUNet"

source "$(dirname "$0")/04_00_common.sh" "$@"
