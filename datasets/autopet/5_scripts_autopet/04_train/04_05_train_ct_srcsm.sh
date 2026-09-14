#!/usr/bin/env bash
# Train srcsm (SRCSM SemRandConv-3D, Thaler et al. 2025) on autopet CT — AugLab-category,
# reuses the auglab_default trainer. 3 folds, 1 GPU/fold, 2000 epochs.
#
# MEASURE srcsm's per-epoch cost on THIS dataset before packing folds on TamIA — CLAUDE.md
# is explicit the ~3x brats/on-harmony slowdown does NOT reliably transfer to a new
# dataset (see [[project_pack_gpu_map]]). Whole-body PET/CT volumes are also far larger
# than any prior dataset here, so re-measure rather than assume ANY prior ratio applies.
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="srcsm"
TRAINER="nnUNetTrainerAutoPETAugLabDefault"
DATASET_ID="120"
DA_WORKERS=8
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_autopet_ct_srcsm"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../00_utils/auglab_configs_fov" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_srcsm_semrandconv.json"
export AUGLAB_VAL_PARAMS_GPU_JSON=""

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
