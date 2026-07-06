#!/usr/bin/env bash
# Train SynthSeg-EM (AugLab + SynthSeg-EM GPU augmentation) on open-ms FLAIR — the MAIN
# CONTENDER. AugLab trainer (nnUNetTrainerOpenMSAugLabDefault) with AUGLAB_PARAMS_GPU_JSON
# pointing at the shared SynthSeg-EM config (label-driven GMM synthesis with EM background
# completion). 4 folds, 1 GPU/fold, 2000 epochs.
#
# Usage:
#   bash 04_02_train_synthseg_EM.sh                                          # auto RUN_ID
#   bash 04_02_train_synthseg_EM.sh open-ms_synthseg_EM_train100_val000_<TS>  # resume
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="synthseg_EM_train100_val000"
TRAINER="nnUNetTrainerOpenMSAugLabDefault"
DATASET_ID="070"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_open-ms_synthseg_EM"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_Synthseg_EM.json"
export AUGLAB_VAL_PARAMS_GPU_JSON=""   # AugLabDefault validates on clean data

# AugLab-category model → co-located under .../<contrast>/auglab.
export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
