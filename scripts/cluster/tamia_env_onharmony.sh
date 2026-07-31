# Source AFTER datasets/on-harmony/5_scripts_on-harmony/00_utils/env.sh (or env_t2w.sh),
# to point on-harmony's OWN training/prediction at scratch-resident data on tamia.
# Mirrors scripts/cluster/tamia_env_chaos.sh's pattern (own-training override). Note
# on-harmony's scratch layout has NO "_on-harmony" suffix on 2_nnUNet/8_results
# (unlike chaos/open-ms) — confirmed against the actual staged directories on tamia.
#   source datasets/on-harmony/5_scripts_on-harmony/00_utils/env_t2w.sh
#   source scripts/cluster/tamia_env_onharmony.sh

export SCRATCH="${SCRATCH:-/scratch/p/paulh}"   # tamia: extra /p/ nesting, unset in non-login shells
ONHARMONY_SCRATCH="$SCRATCH/on-harmony"

export nnUNet_raw="$ONHARMONY_SCRATCH/2_nnUNet/raw"
export nnUNet_preprocessed="$ONHARMONY_SCRATCH/2_nnUNet/preprocessed"
# nnUNet_results/PREDICTIONS_ROOT/METRICS_ROOT: env.sh/env_t2w.sh export these
# UNCONDITIONALLY (same gotcha as chaos' tamia_env_chaos.sh) -- override outright here,
# sourced last.
export nnUNet_results="$ONHARMONY_SCRATCH/8_results/01_predictions/on_harmony_model/${TRAINING_CONTRAST:-T1w}/nnUNet"
export PREDICTIONS_ROOT="$ONHARMONY_SCRATCH/8_results/01_predictions"
export METRICS_ROOT="$ONHARMONY_SCRATCH/8_results/02_metrics"

export RUN_JOB_ACCOUNT="aip-jcohen"
export RUN_JOB_GPU_TYPE="h100"

mkdir -p "$nnUNet_results" "$PREDICTIONS_ROOT" "$METRICS_ROOT"
