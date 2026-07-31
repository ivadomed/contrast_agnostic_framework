# Source AFTER datasets/open-ms/5_scripts_open-ms/00_utils/env.sh (or env_t1w.sh), to
# point open-ms's OWN training/prediction at scratch-resident data on tamia. Mirrors
# scripts/cluster/tamia_env_chaos.sh's pattern (own-training override).
#   source datasets/open-ms/5_scripts_open-ms/00_utils/env_t1w.sh
#   source scripts/cluster/tamia_env_openms.sh

export SCRATCH="${SCRATCH:-/scratch/p/paulh}"   # tamia: extra /p/ nesting, unset in non-login shells
OPENMS_SCRATCH="$SCRATCH/open-ms"

export nnUNet_raw="$OPENMS_SCRATCH/2_nnUNet_open-ms/raw"
export nnUNet_preprocessed="$OPENMS_SCRATCH/2_nnUNet_open-ms/preprocessed"
# nnUNet_results/PREDICTIONS_ROOT/METRICS_ROOT: env.sh/env_t1w.sh export these
# UNCONDITIONALLY (same gotcha as chaos' tamia_env_chaos.sh) -- override outright here,
# sourced last.
export nnUNet_results="$OPENMS_SCRATCH/8_results_open-ms/01_predictions/open_ms_model/${TRAINING_CONTRAST:-flair}/nnUNet"
export PREDICTIONS_ROOT="$OPENMS_SCRATCH/8_results_open-ms/01_predictions"
export METRICS_ROOT="$OPENMS_SCRATCH/8_results_open-ms/02_metrics"

export RUN_JOB_ACCOUNT="aip-jcohen"
export RUN_JOB_GPU_TYPE="h100"

mkdir -p "$nnUNet_results" "$PREDICTIONS_ROOT" "$METRICS_ROOT"
