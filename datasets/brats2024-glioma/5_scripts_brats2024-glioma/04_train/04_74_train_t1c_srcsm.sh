#!/usr/bin/env bash
# Train SRCSM (SemRandConv-3D, Thaler et al. 2025) augmentation on BraTS 2024 Glioma T1c.
# 7th comparison arm (additional to the 6 usual methods); reuses the AugLab-default trainer.
# 3 folds (0 1 2), 1 GPU per fold, 2500 epochs.
#
# Usage:
#   bash 04_74_train_t1c_srcsm.sh           # auto RUN_ID
#   bash 04_74_train_t1c_srcsm.sh brats2024-glioma_t1c_srcsm_<TS>  # resume
# NB: srcsm measured ~3x slower per epoch than the other methods on this dataset's
# other training contrasts (t1n/t2w) — do not assume it will be faster here. A full
# 2500-epoch run will likely need checkpoint-resume (re-run with the same RUN_ID).
# Made overridable so a resume can request a longer wall, e.g.
# RUN_JOB_TIME_DEFAULT=4-00:00:00 bash 04_74_... <RUN_ID>.
export RUN_JOB_TIME_DEFAULT="${RUN_JOB_TIME_DEFAULT:-2-23:00:00}"
source "$(dirname "$0")/../00_utils/env_t1c.sh"

METHOD="srcsm"
TRAINER="nnUNetTrainerBraTS2024GliomaT1cAugLabDefault"
DATASET_ID="054"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_brats2024_t1c_srcsm"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2500}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_srcsm_semrandconv.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
