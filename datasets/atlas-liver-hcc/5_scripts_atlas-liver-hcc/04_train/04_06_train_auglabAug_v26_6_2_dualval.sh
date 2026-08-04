#!/usr/bin/env bash
# OUR METHOD, DUAL VALIDATION on atlas-liver-hcc T1w: ONE training run that produces
# BOTH the clean/val000 and synth-only/val100 "best" checkpoints of the SAME train-time
# config (AugLab default augs + V26_6_2 GPU synthesis @50% train), instead of training
# val000 and val100 separately (half the GPU cost).
#
# Uses nnUNetTrainerAtlasLiverHCCAugLabDualVal (see
# atlas_liver_hcc/trainers/auglab_dualval.py):
#   checkpoint_best.pth         -> clean/val000 best  (this RUN_ID, named _val000_)
#   checkpoint_best_val100.pth  -> synth-only/val100 best
# At on_train_end the trainer hard-links a sibling RUN_ID with "_val000_" swapped
# for "_val100_", so BOTH show up as ordinary predict-ready runs. RUN_ID MUST be
# named "_val000_" (METHOD below ensures this). PREDICT NOTE: predict wrappers for
# BOTH the val000 and val100 RUN_IDs must set
# TRAINER=nnUNetTrainerAtlasLiverHCCAugLabDualVal (both mirrors live under this
# trainer's directory name — NOT AugLabDefault/AugLabValSynth).
#
# RUN_ID contains "auglabAug_v26_6_2_train050_val000" exactly (REQUIRED — this is the
# REF_SUBSTR aggregate_from_config.py keys the auto-wired "sig. vs ref" column off).
# 3 folds (0 1 2), 1 GPU per fold, 2000 epochs.
#
# Usage:
#   bash 04_06_train_auglabAug_v26_6_2_dualval.sh                                              # auto RUN_ID (…_val000_<TS>)
#   bash 04_06_train_auglabAug_v26_6_2_dualval.sh atlas-liver-hcc_t1w_auglabAug_v26_6_2_train050_val000_<TS>  # resume
export RUN_JOB_TIME_DEFAULT="60:00:00"
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerAtlasLiverHCCAugLabDualVal"
DATASET_ID="080"
DA_WORKERS=0
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_atlas-liver-hcc_auglabAug_v26_6_2_train050_dualval"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
