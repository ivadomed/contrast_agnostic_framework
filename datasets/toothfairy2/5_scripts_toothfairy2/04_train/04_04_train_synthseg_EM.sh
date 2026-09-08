#!/usr/bin/env bash
# Train SynthSeg-EM (AugLab + SynthSeg-EM GPU augmentation) on toothfairy2 CBCT —
# historically the strongest rival to OURS on several tasks (closest competitor on
# both BraTS-SSA branches, and the arm liverhccseg had to push past). AugLab trainer
# with the shared SynthSeg-EM config (label-driven GMM synthesis + EM background
# completion). 3 folds, 1 GPU/fold, 1000 epochs.
#
# Usage: bash 04_04_train_synthseg_EM.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="synthseg_EM_train100_val000"
TRAINER="nnUNetTrainerToothFairy2AugLabDefault"
DATASET_ID="${DATASET_ID_CBCT}"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_toothfairy2_cbct_synthseg_EM"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-1000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_Synthseg_EM.json"
export AUGLAB_VAL_PARAMS_GPU_JSON=""   # AugLabDefault validates on clean data

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
