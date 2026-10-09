#!/usr/bin/env bash
# FINAL LADDER RUNG (t1in): the current OURS recipe (04_09_train_auglabAug_v26_6_2_train050_val000.sh: PALETTE + the usual AugLab augmentations,
# val000 trainer) + the boundary partial-volume option (pv_levels in the train config).
# The ONLY diffs vs 04_09_train_auglabAug_v26_6_2_train050_val000.sh: the train config (+ pv keys), METHOD/LOG_DIR naming, chaos T1in epoch default 300->200 (matches its existing OURS run and the chaos policy).
# ALSO removed vs the legacy wrapper (hard-wired to SINGLE_FOLD=0, SINGLE_GPU=3): SINGLE_* pins; nnUNet_compile 1->0 (as the T2spir twin 04_44).
# Generated from that wrapper -- keep in sync.
# AugLab default augmentations + V26_6_2 GPU transform @50% (train), val synth 0%.
# Train synth = config prob 0.5; validation runs on clean data (stock nnUNet
# validation_step). The AugLabV26_6_2 trainer's validation_uses_augmentation=True
# only changes the WandB *viz panel* — actual validation is un-augmented (val 0%).
# 3 folds (0 1 2), 1 GPU per fold (the legacy wrapper was hard-wired to SINGLE_FOLD=0 / GPU 3; removed here).
#
# Usage:
#   bash 04_61_train_t1in_auglabAug_v26_6_2_pv_train050_val000.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="auglabAug_v26_6_2_pv_train050_val000"
TRAINER="nnUNetTrainerCHAOSAugLabV26_6_2"
DATASET_ID="060"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_chaos_auglabAug_v26_6_2_pv_train050_val000"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-200}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglabAug_pv_ImageContrastV26_6_2GPUTransform_train050.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"


source "$(dirname "$0")/04_00_common.sh" "$@"
