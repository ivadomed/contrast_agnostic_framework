#!/usr/bin/env bash
# Train SynthSeg-noEM (AugLab + SynthSeg WITHOUT EM background completion) on
# atlas-liver-hcc T1w — the weaker SynthSeg variant in the standard 6-method suite. Same
# AugLab trainer as auglab_default, AUGLAB_PARAMS_GPU_JSON -> the noEM config.
# 3 folds, 1 GPU/fold, 2000 epochs.
#
# Usage:
#   bash 04_03_train_synthseg_noEM.sh                          # auto RUN_ID
#   bash 04_03_train_synthseg_noEM.sh atlas-liver-hcc_t1w_synthseg_noEM_<TS>   # resume
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="synthseg_noEM"
TRAINER="nnUNetTrainerAtlasLiverHCCAugLabDefault"
DATASET_ID="080"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_atlas-liver-hcc_synthseg_noEM"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_Synthseg.json"
export AUGLAB_VAL_PARAMS_GPU_JSON=""

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
