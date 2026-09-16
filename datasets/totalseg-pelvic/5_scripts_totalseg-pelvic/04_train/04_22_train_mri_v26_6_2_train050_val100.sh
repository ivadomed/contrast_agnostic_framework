#!/usr/bin/env bash
# CAUSAL LADDER (totalseg-pelvic MRI), rung 5 — v26_6_2/PALETTE alone: identical partition to rung 4, REAL-INTENSITY fill. NAMING: METHOD is v26_6_2_train050_val100 with NO auglabAug_ prefix (that prefix denotes rungs 6/7 — see 04_06/04_13's DualVal wrapper) — an auglabAug_-prefixed RUN_ID here would collide with that run's val100 mirror. NNUNET_RESULTS_BASE deliberately NOT set — lands under the default nnUNet-category path, not auglab/.
# 3 folds (0 1 2), 1 GPU/fold, 200 epochs.
source "$(dirname "$0")/../00_utils/env_mri.sh"

METHOD="v26_6_2_train050_val100"
TRAINER="nnUNetTrainerTotalsegPelvicAugLabValSynth"
DATASET_ID="131"
DA_WORKERS=0
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_totalseg_pelvic_mri_v26_6_2_train050_val100"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-200}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_v26_6_2_synth_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"

source "$(dirname "$0")/04_00_common.sh" "$@"
