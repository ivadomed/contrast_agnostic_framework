#!/usr/bin/env bash
# CAUSAL LADDER rung 6 (t1c): rung 5 (04_76_train_t1c_v26_6_2_train050_val100.sh, PALETTE alone, real fill) +
# boundary partial-volume simulation (pv_levels in the train config). The ONLY diff vs
# 04_76_train_t1c_v26_6_2_train050_val100.sh is the train config (+ pv keys) and METHOD/LOG_DIR naming;
# trainer, val config, epochs unchanged. Generated from the rung-5 wrapper -- keep in sync.
# Train V26_6_2 on BraTS 2024 Glioma T1c: 50% train synth / 100% val synth.
# GPU synthesis via nnUNetTrainerBraTS2024GliomaT1cV26_6_2_train050_val100.
# Causal-ablation ladder rung 5 (real-intensity fill; the v26_6_2 alone method,
# identical partition to rung 4/04_59 but real fill instead of noise).
# 3 folds (0 1 2), 1 GPU per fold, 2500 epochs.
#
# Usage:
#   bash 04_84_train_t1c_v26_6_2_pv_train050_val100.sh           # auto RUN_ID
#   bash 04_84_train_t1c_v26_6_2_pv_train050_val100.sh brats2024-glioma_t1c_v26_6_2_train050_val100_<TS>  # resume
export RUN_JOB_TIME_DEFAULT="2-23:00:00"  # 2500 epochs × ~60s/ep ≈ 42h
source "$(dirname "$0")/../00_utils/env_t1c.sh"

METHOD="v26_6_2_pv_train050_val100"
TRAINER="nnUNetTrainerBraTS2024GliomaT1cV26_6_2_train050_val100"
DATASET_ID="054"
DA_WORKERS=16
LOG_DIR="/tmp/nnunet_brats2024_t1c_v26_6_2_pv_train050_val100"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2500}"

# v26_6_2 now runs the AugLab GPU contrast transform (synth + standard spatial DA,
# no other AugLab augs). Both configs required by nnUNetTrainerBraTS2024GliomaT1cV26_6_2*.
_AUGLAB_CONFIGS="$(cd "${PROJECT_ROOT}/sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${_AUGLAB_CONFIGS}/transform_params_gpu_v26_6_2_pv_synth_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${_AUGLAB_CONFIGS}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"
# nnUNet-category run: pin results to the contrast tree (matches predict/eval; on TamIA tamia_env.sh otherwise
# points nnUNet_results at a flat scratch dir and the run must be moved by hand afterwards — found on the T2f port).
export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/nnUNet"

source "$(dirname "$0")/04_00_common.sh" "$@"
