#!/usr/bin/env bash
# Train V26_6_2 ALONE (v26 contrast synthesis + standard spatial DA only — NO full AugLab
# intensity suite) on open-ms FLAIR @50% train, 100% val synth. nnUNet-category model
# (the "Ours, v26 only" row in the 6-method suite). Same trainer class as auglabAug_v26,
# but the v26-only spatialDA config. 4 folds, 1 GPU/fold, 2000 epochs.
#
# Usage:
#   bash 04_06_train_v26_6_2_train050_val100.sh                                        # auto RUN_ID
#   bash 04_06_train_v26_6_2_train050_val100.sh open-ms_flair_v26_6_2_train050_val100_<TS>  # resume
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="v26_6_2_train050_val100"
TRAINER="nnUNetTrainerOpenMSV26_6_2"
DATASET_ID="070"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_open-ms_v26_6_2_train050_val100"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_v26_6_2_synth_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"

# nnUNet-category model → uses env.sh's nnUNet_results default (.../flair/nnUNet).
# No NNUNET_RESULTS_BASE override.

source "$(dirname "$0")/04_00_common.sh" "$@"
