# Source AFTER benchmark/02_tasks/pancreas_disease/pansegdata/5_scripts_pansegdata/00_utils/env.sh (or env_t2w.sh) to point
# pansegdata training/prediction at scratch-resident data on TamIA. Override every path OUTRIGHT (a ${VAR:-} guard here
# is a no-op: common_env.sh has already fired its own guards -- the project notes TamIA gotcha).
#   source benchmark/02_tasks/pancreas_disease/pansegdata/5_scripts_pansegdata/00_utils/env.sh   # or env_t2w.sh
#   source scripts/cluster/tamia_env_pansegdata.sh
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/${USER}}"   # tamia: extra /p/ nesting, unset in non-login shells
PANSEGDATA_SCRATCH="$SCRATCH/pansegdata"

export nnUNet_raw="$PANSEGDATA_SCRATCH/2_nnUNet/raw"
export nnUNet_preprocessed="$PANSEGDATA_SCRATCH/2_nnUNet/preprocessed"
export PREDICTIONS_ROOT="$PANSEGDATA_SCRATCH/8_results/01_predictions"
export nnUNet_results="$PREDICTIONS_ROOT/pansegdata_model/${TRAINING_CONTRAST:-t1wce}/nnUNet"   # same derivation as env.sh
export METRICS_ROOT="$PANSEGDATA_SCRATCH/8_results/02_metrics"
export SPLITS_DIR="$PANSEGDATA_SCRATCH/4_splits"

export RUN_JOB_ACCOUNT="aip-jcohen"
export RUN_JOB_GPU_TYPE="h100"

mkdir -p "$nnUNet_results" "$PREDICTIONS_ROOT" "$METRICS_ROOT"
