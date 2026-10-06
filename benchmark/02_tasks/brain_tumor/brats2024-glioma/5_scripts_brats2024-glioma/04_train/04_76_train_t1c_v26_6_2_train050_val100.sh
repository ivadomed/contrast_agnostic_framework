#!/usr/bin/env bash
# RETIRED 2026-10-05: this trainer validates on SYNTHETIC images (ValSynth: VALsynthonly at p=1), so checkpoint_best is
# chosen on synthetic validation (val100), unlike every other ladder rung. Use the generated *_train050_val000.sh wrapper
# (scripts/cluster/rung5_val000/make_wrappers.py). Set ALLOW_VAL100_ALONE=1 only for a deliberate val100 experiment.
[ "${ALLOW_VAL100_ALONE:-0}" = 1 ] || { echo "ERROR: $(basename "${BASH_SOURCE[0]}") is retired (val100 checkpoint selection); use the *_train050_val000.sh wrapper" >&2; exit 1; }
# Train V26_6_2 on BraTS 2024 Glioma T1c: 50% train synth / 100% val synth.
# GPU synthesis via nnUNetTrainerBraTS2024GliomaT1cV26_6_2_train050_val100.
# Causal-ablation ladder rung 5 (real-intensity fill; the v26_6_2 alone method,
# identical partition to rung 4/04_59 but real fill instead of noise).
# 3 folds (0 1 2), 1 GPU per fold, 2500 epochs.
#
# Usage:
#   bash 04_76_train_t1c_v26_6_2_train050_val100.sh           # auto RUN_ID
#   bash 04_76_train_t1c_v26_6_2_train050_val100.sh brats2024-glioma_t1c_v26_6_2_train050_val100_<TS>  # resume
export RUN_JOB_TIME_DEFAULT="2-23:00:00"  # 2500 epochs × ~60s/ep ≈ 42h
source "$(dirname "$0")/../00_utils/env_t1c.sh"

METHOD="v26_6_2_train050_val100"
TRAINER="nnUNetTrainerBraTS2024GliomaT1cV26_6_2_train050_val100"
DATASET_ID="054"
DA_WORKERS=16
LOG_DIR="/tmp/nnunet_brats2024_t1c_v26_6_2_train050_val100"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2500}"

# v26_6_2 now runs the AugLab GPU contrast transform (synth + standard spatial DA,
# no other AugLab augs). Both configs required by nnUNetTrainerBraTS2024GliomaT1cV26_6_2*.
_AUGLAB_CONFIGS="$(cd "${PROJECT_ROOT}/sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${_AUGLAB_CONFIGS}/transform_params_gpu_v26_6_2_synth_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${_AUGLAB_CONFIGS}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"
# nnUNet-category run: pin results to the contrast tree (matches predict/eval; on TamIA tamia_env.sh otherwise
# points nnUNet_results at a flat scratch dir and the run must be moved by hand afterwards — found on the T2f port).
export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/nnUNet"

source "$(dirname "$0")/04_00_common.sh" "$@"
