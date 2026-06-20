#!/usr/bin/env bash
# Train AugLab with the SynthSeg+EM GPU augmentation config on BraTS 2024 Glioma (T1n).
# Same trainer as 04_05/04_07, but points AUGLAB_PARAMS_GPU_JSON at
# transform_params_gpu_default01-23_Synthseg_EM.json (SynthSeg synthesis WITH EM).
# Uses Dataset051_BraTS2024GliomaT1n (single T1n channel). WandB image panels logged
# automatically by the trainer (_log_wandb_images).
#
# Placement: 4 folds packed onto GPUs 0 and 1 (2 folds per GPU) — folds 0,1 → GPU 0,
# folds 2,3 → GPU 1. GPU is chosen via the physical CUDA_VISIBLE_DEVICES index
# passed through FOLD_SLOT_GPU (Slurm assigns the physical GPU; the index selects within it).
#
# Usage:
#   bash 04_08_train_synthseg_EM.sh                                          # auto: brats2024-glioma_synthseg_EM_train100_val000_<TS>
#   bash 04_08_train_synthseg_EM.sh brats2024-glioma_synthseg_EM_train100_val000_<TS>  # explicit RUN_ID to resume

source "$(dirname "$0")/../00_utils/env.sh"

METHOD="synthseg_EM_train100_val000"
TRAINER="nnUNetTrainerBraTS2024GliomaAugLabDefault"
DATASET_ID="051"
DA_WORKERS=8
LOG_DIR="/tmp/nnunet_brats2024_synthseg_EM"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-1500}"

# SynthSeg + EM augmentation config.
AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_Synthseg_EM.json"

# Save under 01_predictions/auglab (where auglab models live → predict/eval find them).
DATASET_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

# 4 folds → ONLY slots 0 and 1 (GPUs 0,1). 2 folds packed per slot/GPU:
# folds 0,1 → slot 0 / GPU 0 ; folds 2,3 → slot 1 / GPU 1. Slots 2,3 left free.
export FOLD_SLOT_GPU="0,0,0 1,0,0 2,1,1 3,1,1"

source "$(dirname "$0")/04_00_common.sh" "$@"
