# Source AFTER datasets/sliver07/5_scripts_sliver07/00_utils/env.sh, to point
# SLIVER07's cross-dataset (chaos-model) prediction at scratch-resident data on tamia.
#   source datasets/sliver07/5_scripts_sliver07/00_utils/env.sh
#   source scripts/cluster/tamia_env_sliver07_chaoscross.sh
# Cluster differences are expressed as env overrides only — run_job.sh / predict_common.sh
# are never forked. Mirrors scripts/cluster/tamia_env.sh's pattern for brats2024-glioma.

export SCRATCH="${SCRATCH:-/scratch/p/paulh}"   # tamia: extra /p/ nesting, unset in non-login shells

# SLIVER07's own paths (inputs for cross-mode predict + this dataset's own output tree).
export nnUNet_raw="$SCRATCH/sliver07/2_nnUNet_sliver07/raw"
export PREDICTIONS_ROOT="$SCRATCH/sliver07/8_results_sliver07/01_predictions"

# CHAOS_* — model/checkpoint source (env.sh exports these unconditionally, so a
# ${x:-default} fallback never fires; override outright here, same gotcha as
# nnUNet_results in scripts/cluster/tamia_env.sh).
export CHAOS_DATASET_ROOT="$SCRATCH/chaos"
export CHAOS_PREDICTIONS_ROOT="$CHAOS_DATASET_ROOT/8_results_chaos/01_predictions"
export CHAOS_NNUNET_RAW="$CHAOS_DATASET_ROOT/2_nnUNet_chaos/raw"
export CHAOS_NNUNET_PREPROCESSED="$CHAOS_DATASET_ROOT/2_nnUNet_chaos/preprocessed"

# run_job overrides for tamia (whole-node H100, see CLAUDE.md "TamIA" section).
export RUN_JOB_ACCOUNT="aip-jcohen"
export RUN_JOB_GPU_TYPE="h100"

mkdir -p "$PREDICTIONS_ROOT"
