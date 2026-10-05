#!/usr/bin/env bash
# FINAL LADDER RUNG (t1n): the current OURS recipe (04_28_train_t1n_auglabAug_v26_6_2_train050_val000.sh: PALETTE + the usual AugLab augmentations,
# val000 trainer) + the boundary partial-volume option (pv_levels in the train config).
# The ONLY diffs vs 04_28_train_t1n_auglabAug_v26_6_2_train050_val000.sh: the train config (+ pv keys), METHOD/LOG_DIR naming.
# Generated from that wrapper -- keep in sync.
# AugLab augs + V26_6_2 GPU synthesis on BraTS 2024 Glioma T1n: 50% train synth / 0% val
# synth. Train pipeline: full AugLab augs + V26_6_2 synthesis at prob 0.5.
# Val pipeline:   clean (un-augmented), stock nnUNet validation_step.
# 3 folds (0 1 2), 1 GPU per fold, 2500 epochs.
#
# Usage:
#   bash 04_85_train_t1n_auglabAug_v26_6_2_pv_train050_val000.sh           # auto RUN_ID
#   bash 04_85_train_t1n_auglabAug_v26_6_2_pv_train050_val000.sh brats2024-glioma_t1n_auglabAug_v26_6_2_train050_val000_<TS>  # resume
export RUN_JOB_TIME_DEFAULT="2-23:00:00"  # 2500 epochs × ~60s/ep ≈ 42h
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="auglabAug_v26_6_2_pv_train050_val000"
TRAINER="nnUNetTrainerBraTS2024GliomaAugLabDefault"
DATASET_ID="051"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_brats2024_t1n_auglabAug_v26_6_2_pv_train050_val000"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2500}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglabAug_pv_ImageContrastV26_6_2GPUTransform_train050.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
