#!/usr/bin/env bash
# DUAL VALIDATION on ON-Harmony T1w: ONE training run that produces BOTH the
# clean/val000 and synth-only/val100 "best" checkpoints of the SAME train-time config
# (AugLab default augs + V26_6_2 GPU synthesis @50% train), instead of training val000
# and val100 separately. Neither train050_val000 nor train050_val100 has been trained
# for T1w yet (unlike T2w, whose val000 already exists) — this is exactly the case
# DualVal is for: train once, get both.
#
# Uses nnUNetTrainerOnHarmonyAugLabDualVal (see on_harmony/trainers/auglab_dualval.py):
#   checkpoint_best.pth         -> clean/val000 best  (this RUN_ID, named _val000_)
#   checkpoint_best_val100.pth  -> synth-only/val100 best
# At on_train_end the trainer hard-links a sibling RUN_ID with "_val000_" swapped for
# "_val100_", so BOTH show up as ordinary predict-ready runs. RUN_ID MUST be named
# "_val000_" (METHOD below ensures this). PREDICT NOTE: predict wrappers for BOTH the
# val000 and val100 RUN_IDs must set TRAINER=nnUNetTrainerOnHarmonyAugLabDualVal (both
# mirrors live under this trainer's directory name — NOT AugLabDefault/AugLabValSynth).
#
# Usage:
#   bash 04_26_train_t1w_auglabAug_v26_6_2_dualval.sh           # auto RUN_ID (…_val000_<TS>)
#   bash 04_26_train_t1w_auglabAug_v26_6_2_dualval.sh on-harmony_T1w_auglabAug_v26_6_2_train050_val000_<TS>  # resume
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="auglabAug_v26_6_2_train050_val000"
TRAINER="nnUNetTrainerOnHarmonyAugLabDualVal"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_on-harmony_t1w_auglabAug_v26_6_2_dualval"
export NNUNET_NUM_EPOCHS=2000

_AUGLAB_CONFIGS="$(cd "${PROJECT_ROOT}/sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${_AUGLAB_CONFIGS}/transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${_AUGLAB_CONFIGS}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"

# Save under 01_predictions/<contrast>/auglab (auglab-category models → predict/eval find them).
export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
