#!/usr/bin/env bash
# CAUSAL LADDER rung 6 (t1w): rung 5 (04_08_train_t1w_v26_6_2_train050_val100.sh, PALETTE alone, real fill) +
# boundary partial-volume simulation (pv_levels in the train config). The ONLY diff vs
# 04_08_train_t1w_v26_6_2_train050_val100.sh is the train config (+ pv keys) and METHOD/LOG_DIR naming;
# trainer, val config, epochs unchanged. Generated from the rung-5 wrapper -- keep in sync.
# Train V26_6_2 (train_synth_prob=0.50, val_synth_prob=1.0) on ON-Harmony T1w, 2000 epochs.
# Usage: bash 04_47_train_t1w_v26_6_2_pv_train050_val100.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="v26_6_2_pv_train050_val100"
TRAINER="nnUNetTrainerOnHarmonyV26_6_2_train050_val100"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_on-harmony_t1w_v26_6_2_pv_train050_val100"
export NNUNET_NUM_EPOCHS=2000

# v26_6_2 now runs the AugLab GPU contrast transform (synth + standard spatial DA, no
# other AugLab augs). Both configs are required by nnUNetTrainerOnHarmonyAugLabValSynth.
_AUGLAB_CONFIGS="$(cd "${PROJECT_ROOT}/sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${_AUGLAB_CONFIGS}/transform_params_gpu_v26_6_2_pv_synth_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${_AUGLAB_CONFIGS}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"

source "$(dirname "$0")/04_00_common.sh" "$@"
