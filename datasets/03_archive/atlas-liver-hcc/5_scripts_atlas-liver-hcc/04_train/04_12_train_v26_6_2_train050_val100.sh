#!/usr/bin/env bash
# BASELINE-ANCHORED CAUSAL LADDER (atlas-liver-hcc T1w), rung 5 -- v26_6_2 ALONE
# (v26 contrast synthesis + standard spatial DA only, NO full AugLab intensity
# suite), @50% train / 100% val synth. Same shared config already used by chaos's/
# brats2024-glioma's/on-harmony's/open-ms's equivalent rung. This model did NOT
# exist yet for atlas-liver-hcc (only baseline/auglab_default/synthseg/srcsm/
# auglabAug_v26_6_2-dualval existed) -- trained fresh here as the real-fill half
# of the rung 4->5 causal isolation vs 04_11's noise fill (IDENTICAL K-means+
# label-remap+Voronoi partition). Uses nnUNetTrainerAtlasLiverHCCAugLabValSynth
# (synth-only validation, same class already used for parity with the other
# datasets -- previously unused standalone here). nnUNet-category (matches every
# other dataset's rung-5 placement). 3 folds (0 1 2), 1 GPU/fold, 2000 epochs.
#
# Usage: bash 04_12_train_v26_6_2_train050_val100.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="v26_6_2_train050_val100"
TRAINER="nnUNetTrainerAtlasLiverHCCAugLabValSynth"
DATASET_ID="080"
DA_WORKERS=0
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_atlas-liver-hcc_v26_6_2_train050_val100"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_v26_6_2_synth_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"

# nnUNet-category model -> uses env.sh's nnUNet_results default (.../t1w/nnUNet).
# No NNUNET_RESULTS_BASE override (matches every other dataset's rung-5 placement).

source "$(dirname "$0")/04_00_common.sh" "$@"
