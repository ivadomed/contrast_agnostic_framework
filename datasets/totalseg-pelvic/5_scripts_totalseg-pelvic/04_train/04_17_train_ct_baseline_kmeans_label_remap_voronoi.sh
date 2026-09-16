#!/usr/bin/env bash
# CAUSAL LADDER (totalseg-pelvic CT), rung 4 — K-means + label remap + Voronoi sub-parcellation, NOISE FILL. Left side of the key rung4-vs-rung5 one-variable comparison. Do not pre-judge the causal-ablation result here — this is a musculoskeletal (boundary-defined, not appearance/texture-defined) target, closer to chaos organs than open-ms/brats lesions per CLAUDE.md's texture-preservation framing — measure it like every other rung, do not assume the pattern transfers.
# 3 folds (0 1 2), 1 GPU/fold, 200 epochs.
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="baseline_kmeans_label_remap_voronoi"
TRAINER="nnUNetTrainerTotalsegPelvicAugLabDefault"
DATASET_ID="130"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_totalseg_pelvic_ct_baseline_kmeans_label_remap_voronoi"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-200}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_baseline_kmeans_label_remap_voronoi_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON=""

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
