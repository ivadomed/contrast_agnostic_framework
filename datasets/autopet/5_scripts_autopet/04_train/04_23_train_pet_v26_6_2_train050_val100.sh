#!/usr/bin/env bash
# BASELINE-ANCHORED CAUSAL LADDER (autopet PET), rung 5 — "v26_6_2/PALETTE alone":
# identical partition to rung 4, REAL-INTENSITY fill instead of noise.
#
# ⚠️ Same naming trap as the CT version (04_19) — METHOD is "v26_6_2_train050_val100",
# NO "auglabAug_" prefix. Always pass a RUN_ID matching METHOD exactly.
#
# 3 folds (0 1 2), 1 GPU/fold, 2000 epochs.
source "$(dirname "$0")/../00_utils/env_pet.sh"

METHOD="v26_6_2_train050_val100"
TRAINER="nnUNetTrainerAutoPETAugLabValSynth"
DATASET_ID="120"
DA_WORKERS=0
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_autopet_pet_v26_6_2_train050_val100"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../00_utils/auglab_configs_fov" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_v26_6_2_synth_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"

# NNUNET_RESULTS_BASE deliberately NOT set — lands under the default nnUNet path.

source "$(dirname "$0")/04_00_common.sh" "$@"
