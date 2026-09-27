#!/usr/bin/env bash
# LEVEL-3 CAUSAL ABLATION — PALETTE partition + SynthSeg-style Gaussian fill, T1w.
# Identical to 04_12 (auglabAug_v26_6_2, our method) EXCEPT the V26_6_2 contrast transform's
# per-region fill is swapped from real-intensity remap → Gaussian noise (texture destroyed),
# holding the K-means+Voronoi partition fixed. Isolates texture preservation as the causal
# variable. Compare downstream against the already-trained PALETTE arm (04_12).
# Train pipeline: full AugLab augs + V26_6_2-NoiseFill synth at prob 0.25.
# Val pipeline:   synth-only, V26_6_2-NoiseFill at prob 1.0.
# Usage: bash 04_19_train_t1w_auglabAug_v26_6_2_noisefill_train025_val100.sh [RUN_ID]
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="auglabAug_v26_6_2_noisefill_train025_val100"
TRAINER="nnUNetTrainerOnHarmonyAugLabValSynth"
DA_WORKERS=0
LOG_DIR="/tmp/nnunet_on-harmony_t1w_auglabAug_v26_6_2_noisefill_train025_val100"
export NNUNET_NUM_EPOCHS=2000

_AUGLAB_CONFIGS="$(cd "${PROJECT_ROOT}/sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${_AUGLAB_CONFIGS}/transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2NoiseFillGPUTransform_train025.json"
export AUGLAB_VAL_PARAMS_GPU_JSON="${_AUGLAB_CONFIGS}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2NoiseFillGPUTransform.json"

# Save under 01_predictions/<contrast>/auglab (auglab-category models → predict/eval find them).
export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
