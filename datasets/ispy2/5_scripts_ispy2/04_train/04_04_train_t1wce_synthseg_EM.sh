#!/usr/bin/env bash
# Train SynthSeg-EM (AugLab + SynthSeg-EM GPU augmentation) on ispy2 T1WCE — the
# MAIN CONTENDER. AugLab trainer (nnUNetTrainerISPY2AugLabDefault) with
# AUGLAB_PARAMS_GPU_JSON pointing at the shared SynthSeg-EM config (label-driven GMM
# synthesis with EM background completion). 3 folds, 1 GPU/fold, 1000 epochs.
#
# Usage:
#   bash 04_04_train_t1wce_synthseg_EM.sh                                  # auto RUN_ID
#   bash 04_04_train_t1wce_synthseg_EM.sh ispy2_t1wce_synthseg_EM_<TS>     # resume
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="synthseg_EM"
TRAINER="nnUNetTrainerISPY2AugLabDefault"
DATASET_ID="${DATASET_ID_T1WCE}"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_ispy2_t1wce_synthseg_EM"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-1000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_Synthseg_EM.json"
export AUGLAB_VAL_PARAMS_GPU_JSON=""

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
