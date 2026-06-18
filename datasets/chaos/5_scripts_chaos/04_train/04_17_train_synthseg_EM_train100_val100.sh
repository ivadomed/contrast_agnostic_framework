#!/usr/bin/env bash
# SynthSeg_EM with train synth @100% and REAL synth-only validation @100%.
# Folds 0,1 → GPUs 2,3. 200 epochs.
#
# Train pipeline: SynthSeg_EM config (SynthSeg prob 1.0 + light extras).
# Val pipeline:   synth-only, SynthSeg at prob 1.0 (VALsynthonly config) — applied
#                 in a custom validation_step (real val synth, affects metrics).
#
# Usage:
#   bash 04_17_train_synthseg_EM_train100_val100.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="synthseg_EM_train100_val100"
TRAINER="nnUNetTrainerCHAOSAugLabValSynth"
DATASET_ID="060"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_chaos_synthseg_EM_train100_val100"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-200}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_Synthseg_EM.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_Synthseg_EM.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/auglab"

# folds 0,1 → one per GPU (2,3)
export FOLD_SLOT_GPU="${FOLD_SLOT_GPU:-0,2,2 1,3,3}"

source "$(dirname "$0")/04_00_common.sh" "$@"
