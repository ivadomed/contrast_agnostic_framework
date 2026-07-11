#!/usr/bin/env bash
# LEVEL-3 CAUSAL ABLATION v2 — fixes a texture leak in 04_07 (v1): that ablation only
# swapped the K-means/Voronoi region fill to noise, but its per-anatomical-label affine
# remap step was left UNCHANGED (real-texture, fires ~50% directly on the lesion label in
# BOTH arms). v2 (label_fill_noise=true) swaps that step to noise too, so no path in the
# transform preserves real texture — this is the version that actually removes texture
# preservation entirely, isolating it as the causal variable. Compare against 04_03
# (PALETTE) and, if run, 04_09 (kmeans + label remap, no Voronoi).
# 3 folds (0,1,2 — see TRAIN_FOLDS below), 1 GPU/fold, 2000 epochs. AugLabValSynth trainer.
#
# Usage:
#   bash 04_08_train_auglab_kmeans_label_remap_voronoi_train025_val100.sh
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="auglab_kmeans_label_remap_voronoi_train025_val100"
TRAINER="nnUNetTrainerOpenMSAugLabValSynth"
DATASET_ID="070"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_open-ms_auglab_kmeans_label_remap_voronoi_train025_val100"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"
export TRAIN_FOLDS="${TRAIN_FOLDS:-0 1 2}"
# NOTE: unconditional, not ${:-}: env.sh (sourced above) already set
# RUN_JOB_TIME_DEFAULT=60:00:00 (its own ${:-} fallback fires first since the var
# is unset at that point), so a ${RUN_JOB_TIME_DEFAULT:-24:00:00} here would be a
# no-op — the var is already non-empty by the time this line runs.
export RUN_JOB_TIME_DEFAULT=24:00:00

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglab_kmeans_label_remap_voronoi_train025.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_kmeans_label_remap_voronoi.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
