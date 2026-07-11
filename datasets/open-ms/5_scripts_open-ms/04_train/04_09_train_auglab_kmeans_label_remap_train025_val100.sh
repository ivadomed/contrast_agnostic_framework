#!/usr/bin/env bash
# LEVEL-3 CAUSAL ABLATION — isolates the Voronoi spatial sub-parcellation's contribution.
# Identical to 04_08 (kmeans + label remap + Voronoi — texture preservation fully removed)
# EXCEPT skip_sub_parc_prob=1.0: every K-means intensity cluster stays a single region, so
# Voronoi never spatially sub-splits it. K-means partitioning + per-label step remain; only
# the spatial-locality piece is removed. Compare against 04_08 to isolate Voronoi's effect,
# and against 04_03 (PALETTE)/04_02 (SynthSeg-EM) for the full picture.
# 3 folds (0,1,2), 1 GPU/fold, 2000 epochs. AugLabValSynth trainer.
#
# Usage:
#   bash 04_09_train_auglab_kmeans_label_remap_train025_val100.sh
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="auglab_kmeans_label_remap_train025_val100"
TRAINER="nnUNetTrainerOpenMSAugLabValSynth"
DATASET_ID="070"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_open-ms_auglab_kmeans_label_remap_train025_val100"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"
export TRAIN_FOLDS="${TRAIN_FOLDS:-0 1 2}"
# NOTE: unconditional, not ${:-}: env.sh (sourced above) already set
# RUN_JOB_TIME_DEFAULT=60:00:00 (its own ${:-} fallback fires first since the var
# is unset at that point), so a ${RUN_JOB_TIME_DEFAULT:-24:00:00} here would be a
# no-op — the var is already non-empty by the time this line runs.
export RUN_JOB_TIME_DEFAULT=24:00:00

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglab_kmeans_label_remap_train025.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_kmeans_label_remap.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
