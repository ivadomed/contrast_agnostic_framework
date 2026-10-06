#!/usr/bin/env bash
# RETIRED 2026-10-05: this trainer validates on SYNTHETIC images (ValSynth: VALsynthonly at p=1), so checkpoint_best is
# chosen on synthetic validation (val100), unlike every other ladder rung. Use the generated *_train050_val000.sh wrapper
# (scripts/cluster/rung5_val000/make_wrappers.py). Set ALLOW_VAL100_ALONE=1 only for a deliberate val100 experiment.
[ "${ALLOW_VAL100_ALONE:-0}" = 1 ] || { echo "ERROR: $(basename "${BASH_SOURCE[0]}") is retired (val100 checkpoint selection); use the *_train050_val000.sh wrapper" >&2; exit 1; }
# CAUSAL LADDER rung 6 (BraTS T1n): rung 5 (04_20, PALETTE alone, real fill) + boundary
# partial-volume simulation -- K-means thresholds softened in intensity space, Voronoi cuts +
# label-remap edges in a spatial band; one level per sample from pv_levels. The ONLY diff vs
# 04_20 is the pv_* keys in the train config (val config unchanged, so checkpoint selection
# is comparable). 50% train synth / 100% val synth.
# GPU synthesis via nnUNetTrainerBraTS2024GliomaV26_6_2_train050_val100.
# 3 folds (0 1 2, project policy), 1 GPU per fold, 2500 epochs.
#
# Usage:
#   bash 04_81_train_t1n_v26_6_2_pv_train050_val100.sh           # auto RUN_ID
#   bash 04_81_train_t1n_v26_6_2_pv_train050_val100.sh brats2024-glioma_t1n_v26_6_2_pv_train050_val100_<TS>  # resume
export RUN_JOB_TIME_DEFAULT="2-23:00:00"  # 2500 epochs × ~60s/ep ≈ 42h
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="v26_6_2_pv_train050_val100"
TRAINER="nnUNetTrainerBraTS2024GliomaV26_6_2_train050_val100"
DATASET_ID="051"
DA_WORKERS=16
LOG_DIR="/tmp/nnunet_brats2024_t1n_v26_6_2_pv_train050_val100"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2500}"

# v26_6_2 now runs the AugLab GPU contrast transform (synth + standard spatial DA,
# no other AugLab augs). Both configs required by nnUNetTrainerBraTS2024GliomaV26_6_2*.
_AUGLAB_CONFIGS="$(cd "${PROJECT_ROOT}/sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${_AUGLAB_CONFIGS}/transform_params_gpu_v26_6_2_pv_synth_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${_AUGLAB_CONFIGS}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"
source "$(dirname "$0")/04_00_common.sh" "$@"
