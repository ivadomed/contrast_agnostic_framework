#!/usr/bin/env bash
# RETIRED 2026-10-05: this trainer validates on SYNTHETIC images (ValSynth: VALsynthonly at p=1), so checkpoint_best is
# chosen on synthetic validation (val100), unlike every other ladder rung. Use the generated *_train050_val000.sh wrapper
# (scripts/cluster/rung5_val000/make_wrappers.py). Set ALLOW_VAL100_ALONE=1 only for a deliberate val100 experiment.
[ "${ALLOW_VAL100_ALONE:-0}" = 1 ] || { echo "ERROR: $(basename "${BASH_SOURCE[0]}") is retired (val100 checkpoint selection); use the *_train050_val000.sh wrapper" >&2; exit 1; }
# Train V26_6_2 (train_synth_prob=0.50, val_synth_prob=1.0) on ON-Harmony T1w, 2000 epochs.
# Usage: bash 04_08_train_t1w_v26_6_2_train050_val100.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="v26_6_2_train050_val100"
TRAINER="nnUNetTrainerOnHarmonyV26_6_2_train050_val100"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_on-harmony_t1w_v26_6_2_train050_val100"
export NNUNET_NUM_EPOCHS=2000

# v26_6_2 now runs the AugLab GPU contrast transform (synth + standard spatial DA, no
# other AugLab augs). Both configs are required by nnUNetTrainerOnHarmonyAugLabValSynth.
_AUGLAB_CONFIGS="$(cd "${PROJECT_ROOT}/sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${_AUGLAB_CONFIGS}/transform_params_gpu_v26_6_2_synth_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${_AUGLAB_CONFIGS}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"

source "$(dirname "$0")/04_00_common.sh" "$@"
