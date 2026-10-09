# Source AFTER benchmark/02_tasks/brain_stroke/isles2022/5_scripts_isles2022/00_utils/env.sh (or env_flair.sh) to point
# isles2022 training/prediction at scratch-resident data on TamIA. Override every path OUTRIGHT (a ${VAR:-} guard here
# is a no-op: common_env.sh has already fired its own guards -- the project notes TamIA gotcha).
#   source benchmark/02_tasks/brain_stroke/isles2022/5_scripts_isles2022/00_utils/env.sh   # or env_flair.sh
#   source scripts/cluster/tamia_env_isles2022.sh
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/${USER}}"   # tamia: extra /p/ nesting, unset in non-login shells
ISLES2022_SCRATCH="$SCRATCH/isles2022"

export nnUNet_raw="$ISLES2022_SCRATCH/2_nnUNet/raw"
export nnUNet_preprocessed="$ISLES2022_SCRATCH/2_nnUNet/preprocessed"
export PREDICTIONS_ROOT="$ISLES2022_SCRATCH/8_results/01_predictions"
export nnUNet_results="$PREDICTIONS_ROOT/isles2022_model/${TRAINING_CONTRAST:-dwi}/nnUNet"   # same derivation as env.sh
export METRICS_ROOT="$ISLES2022_SCRATCH/8_results/02_metrics"
export SPLITS_DIR="$ISLES2022_SCRATCH/4_splits"

export RUN_JOB_ACCOUNT="aip-jcohen"
export RUN_JOB_GPU_TYPE="h100"

mkdir -p "$nnUNet_results" "$PREDICTIONS_ROOT" "$METRICS_ROOT"
