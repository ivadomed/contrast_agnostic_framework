# Source AFTER datasets/chaos/5_scripts_chaos/00_utils/env.sh (or env_t2spir.sh), to
# point CHAOS's OWN training/prediction at scratch-resident data on tamia. Mirrors
# scripts/cluster/tamia_env.sh's pattern for brats2024-glioma (own-training override,
# distinct from tamia_env_{amos,sliver07}_chaoscross.sh which point the OTHER
# direction -- amos/sliver07 predicting with CHAOS-trained checkpoints).
#   source datasets/chaos/5_scripts_chaos/00_utils/env_t2spir.sh
#   source scripts/cluster/tamia_env_chaos.sh

export SCRATCH="${SCRATCH:-/scratch/p/paulh}"   # tamia: extra /p/ nesting, unset in non-login shells
CHAOS_SCRATCH="$SCRATCH/chaos"

export nnUNet_raw="$CHAOS_SCRATCH/2_nnUNet_chaos/raw"
export nnUNet_preprocessed="$CHAOS_SCRATCH/2_nnUNet_chaos/preprocessed"
# nnUNet_results/PREDICTIONS_ROOT: env.sh/env_t2spir.sh export these UNCONDITIONALLY
# (same gotcha as brats' tamia_env.sh) -- override outright here, sourced last.
export nnUNet_results="$CHAOS_SCRATCH/8_results_chaos/01_predictions/chaos_model/${TRAINING_CONTRAST:-t1in}/nnUNet"
export PREDICTIONS_ROOT="$CHAOS_SCRATCH/8_results_chaos/01_predictions"
export METRICS_ROOT="$CHAOS_SCRATCH/8_results_chaos/02_metrics"

export RUN_JOB_ACCOUNT="aip-jcohen"
export RUN_JOB_GPU_TYPE="h100"

mkdir -p "$nnUNet_results" "$PREDICTIONS_ROOT" "$METRICS_ROOT"
