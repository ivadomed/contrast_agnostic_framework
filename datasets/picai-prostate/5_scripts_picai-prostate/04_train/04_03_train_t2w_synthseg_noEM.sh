#!/usr/bin/env bash
# Train SYNTHSEG_noEM (SynthSeg-style label-driven synthesis, no expectation-maximisation step) on picai-prostate T2W (Dataset080_PICAI_T2W).
# csPCa lesion segmentation; 3 folds (0 1 2), 1 GPU/fold, 2000 epochs.
#
# Usage:
#   bash 04_03_train_t2w_synthseg_noEM.sh                                  # auto RUN_ID
#   bash 04_03_train_t2w_synthseg_noEM.sh picai-prostate_t2w_synthseg_noEM_<TS>   # resume
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="synthseg_noEM"
TRAINER="nnUNetTrainerPICAIProstateAugLabDefault"
CATEGORY="auglab"
DATASET_ID="080"
DA_WORKERS=6
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_Synthseg.json"
export AUGLAB_VAL_PARAMS_GPU_JSON=""

source "$(dirname "$0")/04_00_common.sh" "$@"
