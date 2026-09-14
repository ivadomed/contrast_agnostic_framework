#!/usr/bin/env bash
# BASELINE-ANCHORED CAUSAL LADDER (autopet CT), rung 5 — "v26_6_2/PALETTE alone":
# identical partition to rung 4, REAL-INTENSITY fill instead of noise. The causal pair
# against rung 4.
#
# ⚠️ NAMING TRAP (cost a cancel-and-resubmit on the ambl run — see CLAUDE.md): METHOD
# here is "v26_6_2_train050_val100" — NO "auglabAug_" prefix. That prefix denotes rungs
# 6/7 (PALETTE + AugLab, see 04_06). An "auglabAug_"-prefixed RUN_ID here would collide
# with the DualVal val100 mirror from the rung-6 run and destroy this rung's own
# comparison. Always pass a RUN_ID matching METHOD exactly.
#
# Trainer is ValSynth (not Baseline/AugLabDefault/DualVal) — it validates on synth-only
# data, matching what rung 6/7's DualVal run selects as ITS val100 checkpoint, so this
# standalone rung is selection-equivalent to that mirror rather than an approximation.
#
# 3 folds (0 1 2), 1 GPU/fold, 2000 epochs.
#
# Usage: bash 04_19_train_ct_v26_6_2_train050_val100.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="v26_6_2_train050_val100"
TRAINER="nnUNetTrainerAutoPETAugLabValSynth"
DATASET_ID="120"
DA_WORKERS=0
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_autopet_ct_v26_6_2_train050_val100"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../00_utils/auglab_configs_fov" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_v26_6_2_synth_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"

# NNUNET_RESULTS_BASE deliberately NOT set (matches every other dataset's rung 5
# exactly) — lands under the default nnUNet-category path, not auglab/.

source "$(dirname "$0")/04_00_common.sh" "$@"
