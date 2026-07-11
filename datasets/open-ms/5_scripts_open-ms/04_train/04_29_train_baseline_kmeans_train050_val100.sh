#!/usr/bin/env bash
# BASELINE-ANCHORED CAUSAL LADDER, new rung between baseline (floor) and
# baseline_kmeans_label_remap — K-means intensity clustering + noise fill ONLY,
# no per-anatomical-label remap (label_remap_prob=0.0, everything else identical
# to 04_10: no Voronoi, skip_sub_parc_prob=1.0). Isolates the K-means partition's
# own contribution before the per-label remap step is added.
# Ladder: baseline (04_01) -> THIS -> +label remap (04_10) -> +Voronoi (04_11)
# -> v26_6_2 alone (04_06). 3 folds (0,1,2), 1 GPU/fold, 2000 epochs.
# nnUNet-category, AugLabValSynth trainer (synth-only validation, matches 04_10).
#
# Usage:
#   bash 04_29_train_baseline_kmeans_train050_val100.sh
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="baseline_kmeans_train050_val100"
TRAINER="nnUNetTrainerOpenMSAugLabValSynth"
DATASET_ID="070"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_open-ms_baseline_kmeans_train050_val100"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"
export TRAIN_FOLDS="${TRAIN_FOLDS:-0 1 2}"
export RUN_JOB_TIME_DEFAULT=24:00:00

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_baseline_kmeans_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_kmeans.json"

# nnUNet-category model → uses env.sh's nnUNet_results default (.../flair/nnUNet),
# same placement as 04_10/04_11. No NNUNET_RESULTS_BASE override.

source "$(dirname "$0")/04_00_common.sh" "$@"
