#!/usr/bin/env bash
# OUR METHOD, DUAL VALIDATION on BraTS 2024 Glioma T2w: ONE training run that
# produces BOTH the clean/val000 and synth-only/val100 "best" checkpoints of the
# SAME train-time config (AugLab default augs + V26_6_2 GPU synthesis @50% train),
# instead of training val000 and val100 separately (half the GPU cost).
#
# Uses nnUNetTrainerBraTS2024GliomaT2wAugLabDualVal (see
# brats2024_glioma/trainers/auglab_dualval.py):
#   checkpoint_best.pth         -> clean/val000 best  (this RUN_ID, named _val000_)
#   checkpoint_best_val100.pth  -> synth-only/val100 best
# At on_train_end the trainer hard-links a sibling RUN_ID with "_val000_" swapped
# for "_val100_", so BOTH show up as ordinary predict-ready runs. RUN_ID MUST be
# named "_val000_" (METHOD below ensures this). PREDICT NOTE: predict wrappers for
# BOTH the val000 and val100 RUN_IDs must set
# TRAINER=nnUNetTrainerBraTS2024GliomaT2wAugLabDualVal (both mirrors live under
# this trainer's directory name — NOT AugLabDefault/AugLabValSynth).
#
# Supersedes the separate 04_29 (val000) + 04_31 (val100) pair for T2w.
# 3 folds (0 1 2), 1 GPU per fold, 2500 epochs.
#
# Usage:
#   bash 04_32_train_t2w_auglabAug_v26_6_2_dualval.sh           # auto RUN_ID (…_val000_<TS>)
#   bash 04_32_train_t2w_auglabAug_v26_6_2_dualval.sh brats2024-glioma_t2w_auglabAug_v26_6_2_train050_val000_<TS>  # resume
export RUN_JOB_TIME_DEFAULT="2-23:00:00"  # 2500 epochs x ~60s/ep ~= 42h
source "$(dirname "$0")/../00_utils/env_t2w.sh"

METHOD="auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerBraTS2024GliomaT2wAugLabDualVal"
DATASET_ID="052"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_brats2024_t2w_auglabAug_v26_6_2_train050_dualval"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2500}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
