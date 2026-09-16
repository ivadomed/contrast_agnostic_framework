#!/usr/bin/env bash
# CAUSAL LADDER (totalseg-pelvic MRI), rung 3 — K-means + label remap, still noise fill, no Voronoi.
# 3 folds (0 1 2), 1 GPU/fold, 200 epochs.
source "$(dirname "$0")/../00_utils/env_mri.sh"

METHOD="baseline_kmeans_label_remap"
TRAINER="nnUNetTrainerTotalsegPelvicAugLabDefault"
DATASET_ID="131"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_totalseg_pelvic_mri_baseline_kmeans_label_remap"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-200}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_baseline_kmeans_label_remap_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON=""

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
