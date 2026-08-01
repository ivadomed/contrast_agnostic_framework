# Source AFTER datasets/kidney-t2w/5_scripts_kidney-t2w/00_utils/env.sh, to
# point KIDNEY-T2W's cross-dataset (chaos-model) prediction at scratch-resident
# data on tamia.
#   source datasets/kidney-t2w/5_scripts_kidney-t2w/00_utils/env.sh
#   source scripts/cluster/tamia_env_kidney-t2w_chaoscross.sh
# Cluster differences are expressed as env overrides only — run_job.sh / predict_common.sh
# are never forked. Mirrors scripts/cluster/tamia_env_cirrmri-liver_chaoscross.sh's pattern.

export SCRATCH="${SCRATCH:-/scratch/p/paulh}"   # tamia: extra /p/ nesting, unset in non-login shells

# KIDNEY-T2W's own paths (inputs for cross-mode predict + this dataset's own output tree).
# METRICS_ROOT is intentionally left at common_env.sh's default (under $PROJECT, NOT
# scratch) — prediction NIfTI volumes are large/regenerable (scratch, purge-able),
# but the small per-fold metrics CSVs are precious and land in the backed-up
# $PROJECT repo checkout so they can be copied back to Vulcan afterward.
export nnUNet_raw="$SCRATCH/kidney-t2w/2_nnUNet_kidney-t2w/raw"
export PREDICTIONS_ROOT="$SCRATCH/kidney-t2w/8_results_kidney-t2w/01_predictions"

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
