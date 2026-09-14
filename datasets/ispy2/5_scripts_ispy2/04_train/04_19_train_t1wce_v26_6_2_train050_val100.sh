#!/usr/bin/env bash
# BASELINE-ANCHORED CAUSAL LADDER (ispy2 T1WCE), rung 5 -- "v26_6_2/PALETTE alone"
# (identical partition to rung 4, REAL-INTENSITY fill instead of noise). This is
# the causal pair vs rung 4 — the key one-variable test of whether texture
# preservation drives Dice.
#
# ⚠️ CRITICAL NAMING TRAP (cost a cancel-and-resubmit on the ambl run): this
# script's METHOD is "v26_6_2_train050_val100" — NO "auglabAug_" prefix. That
# prefix denotes rungs 6/7 (PALETTE + AugLab, see 04_06/04_13). Always pass this
# script a RUN_ID matching METHOD exactly (e.g. ispy2_t1wce_v26_6_2_train050_val100_<TS>)
# — an "auglabAug_"-prefixed RUN_ID here would collide with the DualVal val100
# mirror from the rung-6 run and destroy this rung's own comparison.
#
# Trainer is ValSynth (not Baseline/AugLabDefault/DualVal) — see
# ispy2/trainers/auglab_valsynth.py: validates on synth-only data, matching what
# rung 6/7's DualVal run selects as ITS val100 checkpoint, so this standalone rung
# is selection-equivalent to that mirror rather than an approximation of it.
#
# 3 folds (0 1 2), 1 GPU/fold, 1000 epochs.
#
# Usage: bash 04_19_train_t1wce_v26_6_2_train050_val100.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="v26_6_2_train050_val100"
TRAINER="nnUNetTrainerISPY2AugLabValSynth"
DATASET_ID="${DATASET_ID_T1WCE}"
DA_WORKERS=0
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_ispy2_t1wce_v26_6_2_train050_val100"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-1000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_v26_6_2_synth_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"

# nnUNET_RESULTS_BASE deliberately NOT set here (matches ambl's 04_19 exactly) —
# this rung lands under the default nnUNet-category path, not auglab/. This is a
# real, existing quirk carried forward for consistency with ambl's layout, not a
# new decision.

source "$(dirname "$0")/04_00_common.sh" "$@"
