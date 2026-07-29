# Source AFTER datasets/brats-ssa2024/5_scripts_brats-ssa2024/00_utils/env.sh (or env_t2w.sh),
# to point brats-ssa2024's cross-dataset (brats2024-glioma-model) prediction at
# scratch-resident data on tamia.
#   source datasets/brats-ssa2024/5_scripts_brats-ssa2024/00_utils/env.sh
#   source scripts/cluster/tamia_env_brats-ssa2024_bratscross.sh
# Cluster differences are expressed as env overrides only -- run_job.sh / predict_common.sh
# are never forked. Mirrors tamia_env_mslesseg_openmscross.sh's pattern (same section-14
# cross-dataset shape, BRATS_* instead of OPENMS_*).
#
# brats-ssa2024's OWN nnUNet_raw needs NO override here -- 2_nnUNet_brats-ssa2024/ is
# already a symlink into $SCRATCH/brats-ssa2024/ (set up when the raw data was staged),
# so common_env.sh's DATASET_ROOT-relative path already resolves onto scratch. Only
# PREDICTIONS_ROOT/METRICS_ROOT (which would otherwise land under the git-repo path's
# 8_results_brats-ssa2024/, burning tamia's tight $PROJECT file-count quota across
# 12 methods x 3 folds x 4 contrasts x 95 cases) and the BRATS_* source-model vars
# need overriding.

export SCRATCH="${SCRATCH:-/scratch/p/paulh}"   # tamia: extra /p/ nesting, unset in non-login shells

export PREDICTIONS_ROOT="$SCRATCH/brats-ssa2024/8_results_brats-ssa2024/01_predictions"
export METRICS_ROOT="$SCRATCH/brats-ssa2024/8_results_brats-ssa2024/02_metrics"

# BRATS_* -- model/checkpoint source (env.sh exports these unconditionally... no wait,
# guarded with ${:-} already -- override outright here since this file always wins by
# being sourced last).
export BRATS_DATASET_ROOT="$SCRATCH/brats2024-glioma"
export BRATS_PREDICTIONS_ROOT="$BRATS_DATASET_ROOT/8_results/01_predictions"
export BRATS_NNUNET_RAW="$BRATS_DATASET_ROOT/2_nnUNet/raw"
export BRATS_NNUNET_PREPROCESSED="$BRATS_DATASET_ROOT/2_nnUNet/preprocessed"
# BRATS_DATASET_ID/_DS_NAME/_TRAINING_CONTRAST: leave at whatever env.sh/env_t2w.sh already
# set (t1n/051 by default, t2w/052 if env_t2w.sh was sourced first) -- do NOT override here.

# run_job overrides for tamia (whole-node H100, see CLAUDE.md "TamIA" section).
export RUN_JOB_ACCOUNT="aip-jcohen"
export RUN_JOB_GPU_TYPE="h100"

mkdir -p "$PREDICTIONS_ROOT" "$METRICS_ROOT"
