#!/usr/bin/env bash
# Train SynthSeg (no EM) GPU augmentation on CHAOS MR T2spir (no AugLab aug, synth prob=1).
# Uses transform_params_gpu_default01-23_Synthseg.json — AugLab augs are deactivated
# (prob=0 in this config); only SynthSeg-style synthesis without EM is active.
# 4 folds, 1 GPU per fold, 200 epochs.
#
# Usage:
#   bash 04_31_train_t2spir_synthseg_noEM.sh                               # auto RUN_ID
#   bash 04_31_train_t2spir_synthseg_noEM.sh chaos_t2spir_synthseg_noEM_<TS>  # resume
source "$(dirname "$0")/../00_utils/env_t2spir.sh"

METHOD="synthseg_noEM"
TRAINER="nnUNetTrainerCHAOSAugLabDefault"
DATASET_ID="061"
DA_WORKERS="${DA_WORKERS:-0}"
LOG_DIR="/tmp/nnunet_chaos_t2spir_synthseg_noEM"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-200}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_Synthseg.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
