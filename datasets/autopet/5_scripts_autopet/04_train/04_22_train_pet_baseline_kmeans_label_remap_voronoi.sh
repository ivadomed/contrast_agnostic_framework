#!/usr/bin/env bash
# BASELINE-ANCHORED CAUSAL LADDER (autopet PET), rung 4 — K-means + label remap +
# Voronoi sub-parcellation, NOISE FILL. Causal pair against rung 5 (real-fill).
# 3 folds (0 1 2), 1 GPU/fold, 2000 epochs.
source "$(dirname "$0")/../00_utils/env_pet.sh"

METHOD="baseline_kmeans_label_remap_voronoi"
TRAINER="nnUNetTrainerAutoPETAugLabDefault"
DATASET_ID="120"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_autopet_pet_baseline_kmeans_label_remap_voronoi"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../00_utils/auglab_configs_fov" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_baseline_kmeans_label_remap_voronoi_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON=""

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
