#!/usr/bin/env bash
# Train SynthSeg without EM (uniform intensity priors) on ON-Harmony T2w, 2000 epochs.
# Uses nnUNetTrainerOnHarmonyAugLabDefault + Synthseg (no EM) AugLab config —
# same approach as brats2024-glioma / CHAOS (not BrainGenerator).
# Usage: bash 04_16_train_t2w_synthseg_noEM.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env_t2w.sh"
METHOD="synthseg_noEM"
TRAINER="nnUNetTrainerOnHarmonyAugLabDefault"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_on-harmony_t2w_synthseg_noEM"
export DATASET_ID="032"
export NNUNET_NUM_EPOCHS=2000

_AUGLAB_CONFIGS="$(cd "${PROJECT_ROOT}/sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${_AUGLAB_CONFIGS}/transform_params_gpu_default01-23_Synthseg.json"

# Save under 01_predictions/<contrast>/auglab (auglab-category models → predict/eval find them).
export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
