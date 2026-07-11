#!/usr/bin/env bash
# AUGLAB-ANCHORED CAUSAL LADDER, new rung between auglab_default (floor) and
# auglab_kmeans_label_remap — K-means intensity clustering + noise fill ONLY,
# no per-anatomical-label remap (label_remap_prob=0.0, everything else identical
# to 04_09: no Voronoi, skip_sub_parc_prob=1.0, full AugLab default01-23 suite
# held fixed). Isolates the K-means partition's own contribution before the
# per-label remap step is added.
# Ladder: auglab_default (04_04) -> THIS -> +label remap (04_09) -> +Voronoi
# (04_08) -> auglabAug_v26_6_2 (04_03). 3 folds (0,1,2), 1 GPU/fold, 2000 epochs.
# AugLabValSynth trainer (synth-only validation, matches 04_09).
#
# Usage:
#   bash 04_31_train_auglab_kmeans_train025_val100.sh
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="auglab_kmeans_train025_val100"
TRAINER="nnUNetTrainerOpenMSAugLabValSynth"
DATASET_ID="070"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_open-ms_auglab_kmeans_train025_val100"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"
export TRAIN_FOLDS="${TRAIN_FOLDS:-0 1 2}"
export RUN_JOB_TIME_DEFAULT=24:00:00

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglab_kmeans_train025.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_kmeans.json"

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
