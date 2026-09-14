#!/usr/bin/env bash
# BASELINE-ANCHORED CAUSAL LADDER (autopet CT), rung 4 — K-means + label remap +
# Voronoi sub-parcellation, NOISE FILL. This is the left-hand side of the ladder's key
# one-variable comparison: rung 4 (noise fill) vs rung 5 (real-intensity fill) share an
# identical partition and differ ONLY in what fills it.
#
# The paper's prediction for AutoPET's appearance-defined (PET uptake / CT enhancement
# pattern) tumor-lesion target is a LARGE 4->5 gain, same pattern as open-ms lesions /
# brats tumour sub-regions — this dataset groups with those, not with chaos organs. Do
# not pre-judge it — measure it like every other rung.
#
# 3 folds (0 1 2), 1 GPU/fold, 2000 epochs.
#
# Usage: bash 04_18_train_ct_baseline_kmeans_label_remap_voronoi.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="baseline_kmeans_label_remap_voronoi"
TRAINER="nnUNetTrainerAutoPETAugLabDefault"
DATASET_ID="120"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_autopet_ct_baseline_kmeans_label_remap_voronoi"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../00_utils/auglab_configs_fov" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_baseline_kmeans_label_remap_voronoi_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON=""

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
