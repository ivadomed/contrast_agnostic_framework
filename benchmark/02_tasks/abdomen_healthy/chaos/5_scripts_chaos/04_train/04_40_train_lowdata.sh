#!/usr/bin/env bash
# Low-data-regime benchmark — train ONE method at ONE regime N on CHAOS MR T1in.
#
# ADDITIVE: reuses each method's exact config but swaps in the *LowData trainer
# variant (LowDataMixin, see CHAOSLowDataTrainers.py + lowdata_mixin.py). The regime
# N is carried in the RUN_ID (`_lowdata_n<NN>`), which the driver already exports into
# the job; the mixin reads it and keeps the first N train subjects per fold (val
# untouched). Existing full-data runs are unaffected.
#
# Usage:
#   bash 04_40_train_lowdata.sh <method> <N> [RUN_ID]
#     <method> ∈ baseline | v26_6_2 | auglab_default | synthseg_EM | synthseg_noEM | auglabAug_v26_6_2
#     <N>      number of train subjects/fold (1 2 4 8 12). 12 ≈ full (MR-CV pool is ~12/fold).
#     [RUN_ID] optional, to RESUME an existing run.
#
# Epochs are held FIXED across regimes (fair comparison — nnU-Net epochs are fixed
# iterations, independent of N). Override the whole sweep via NNUNET_NUM_EPOCHS.
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"

METHOD_KEY="${1:?method required (baseline|v26_6_2|auglab_default|synthseg_EM|synthseg_noEM|auglabAug_v26_6_2)}"
N="${2:?N required (e.g. 1 2 4 8 12)}"
RESUME_RUN_ID="${3:-}"

DATASET_ID="060"
export nnUNet_compile=0
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-200}"    # CHAOS standard, fixed across all regimes
AUGLAB_CONFIGS_DIR="${PROJECT_ROOT}/sub-workspaces/auglab_workspace/AugLab/auglab/configs"

case "${METHOD_KEY}" in
  baseline)
    METHOD="baseline"
    TRAINER="nnUNetTrainerCHAOSBaselineLowData"
    DA_WORKERS="${DA_WORKERS:-16}"
    ;;
  v26_6_2)
    METHOD="v26_6_2_train050_val100"
    TRAINER="nnUNetTrainerCHAOSV26_6_2_p50LowData"
    DA_WORKERS="${DA_WORKERS:-0}"
    # v26_6_2 now runs the AugLab GPU contrast transform (synth + spatial DA, no other
    # AugLab augs); the trainer requires both configs. Results stay in nnUNet/ category
    # (no NNUNET_RESULTS_BASE override), matching prior v26_6_2 runs.
    export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_v26_6_2_synth_spatialDA_train050.json"
    export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"
    ;;
  auglab_default)
    METHOD="auglab_default"
    TRAINER="nnUNetTrainerCHAOSAugLabDefaultLowData"
    DA_WORKERS="${DA_WORKERS:-0}"
    export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23.json"
    export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"
    ;;
  synthseg_EM)
    METHOD="synthseg_EM"
    TRAINER="nnUNetTrainerCHAOSAugLabDefaultLowData"
    DA_WORKERS="${DA_WORKERS:-0}"
    export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_Synthseg_EM.json"
    export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"
    ;;
  synthseg_noEM)
    METHOD="synthseg_noEM"
    TRAINER="nnUNetTrainerCHAOSAugLabDefaultLowData"
    DA_WORKERS="${DA_WORKERS:-0}"
    export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_Synthseg.json"
    export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"
    ;;
  auglabAug_v26_6_2)
    METHOD="auglabAug_v26_6_2_train025_val100"
    TRAINER="nnUNetTrainerCHAOSAugLabValSynthLowData"
    DA_WORKERS="${DA_WORKERS:-0}"
    export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train025.json"
    export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"
    export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"
    ;;
  *)
    echo "ERROR: unknown method '${METHOD_KEY}'" >&2; exit 1 ;;
esac

NN="$(printf '%02d' "${N}")"
LOG_DIR="/tmp/nnunet_chaos_lowdata_${METHOD_KEY}_n${NN}"

if [ -n "${RESUME_RUN_ID}" ]; then
    RUN_ID="${RESUME_RUN_ID}"
else
    RUN_ID="chaos_${TRAINING_CONTRAST}_${METHOD}_lowdata_n${NN}_$(date +%Y%m%d_%H%M%S)"
fi

# 3 h allocation per fold — CHAOS trains well under that even for synth methods
# (~40s/epoch baseline → a few × that for synthesis; 200 epochs ≪ 180 min), and the
# shorter ask schedules faster. Each fold is its own job.
export RUN_JOB_TIME_DEFAULT="${RUN_JOB_TIME_DEFAULT:-3:00:00}"

echo "[lowdata] method=${METHOD_KEY} N=${N} trainer=${TRAINER} epochs=${NNUNET_NUM_EPOCHS}"
source "$(dirname "$0")/04_00_common.sh" "${RUN_ID}"
