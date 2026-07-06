#!/usr/bin/env bash
# Train SynthSeg-noEM (AugLab + SynthSeg WITHOUT EM background completion) on open-ms
# FLAIR — the weaker SynthSeg variant in the standard 6-method suite. Same AugLab trainer
# as synthseg_EM, AUGLAB_PARAMS_GPU_JSON → the noEM config. 4 folds, 1 GPU/fold, 2000 ep.
#
# Usage:
#   bash 04_05_train_synthseg_noEM.sh                                          # auto RUN_ID
#   bash 04_05_train_synthseg_noEM.sh open-ms_flair_synthseg_noEM_train100_val000_<TS>  # resume
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="synthseg_noEM_train100_val000"
TRAINER="nnUNetTrainerOpenMSAugLabDefault"
DATASET_ID="070"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_open-ms_synthseg_noEM"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_Synthseg.json"
export AUGLAB_VAL_PARAMS_GPU_JSON=""

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
