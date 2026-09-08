#!/usr/bin/env bash
# BASELINE-ANCHORED CAUSAL LADDER (toothfairy2 CBCT), rung 4 — K-means + label
# remap + Voronoi sub-parcellation, NOISE FILL. This is the left-hand side of the
# ladder's key one-variable comparison: rung 4 (noise fill) vs rung 5 (real-intensity
# fill) share an identical partition and differ ONLY in what fills it.
#
# The paper's prediction for a BOUNDARY-DEFINED target like this one is that the
# 4->5 step is near-zero (as on chaos organs), in contrast to the large gain on
# texture-defined targets (open-ms lesions, brats tumour sub-regions). That
# prediction is the reason this dataset was onboarded — do not pre-judge it.
#
# 3 folds (0 1 2), 1 GPU/fold, 1000 epochs.
#
# Usage: bash 04_10_train_baseline_kmeans_label_remap_voronoi.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="baseline_kmeans_label_remap_voronoi"
TRAINER="nnUNetTrainerToothFairy2AugLabDefault"
DATASET_ID="${DATASET_ID_CBCT}"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_toothfairy2_cbct_baseline_kmeans_label_remap_voronoi"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-1000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_baseline_kmeans_label_remap_voronoi_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON=""

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
