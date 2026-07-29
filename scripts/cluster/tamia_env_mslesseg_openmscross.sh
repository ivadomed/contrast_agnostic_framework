# Source AFTER datasets/mslesseg/5_scripts_mslesseg/00_utils/env.sh, to point MSLesSeg's
# cross-dataset (open-ms-model) prediction at scratch-resident data on tamia.
#   source datasets/mslesseg/5_scripts_mslesseg/00_utils/env.sh
#   source scripts/cluster/tamia_env_mslesseg_openmscross.sh
# Cluster differences are expressed as env overrides only -- run_job.sh / predict_common.sh
# are never forked. Mirrors tamia_env_{amos,sliver07}_chaoscross.sh's pattern (same
# section-14 cross-dataset shape, just OPENMS_* instead of CHAOS_*).

export SCRATCH="${SCRATCH:-/scratch/p/paulh}"   # tamia: extra /p/ nesting, unset in non-login shells

# MSLesSeg's own paths (inputs for cross-mode predict + this dataset's own output tree).
export nnUNet_raw="$SCRATCH/mslesseg/2_nnUNet_mslesseg/raw"
export PREDICTIONS_ROOT="$SCRATCH/mslesseg/8_results_mslesseg/01_predictions"
export METRICS_ROOT="$SCRATCH/mslesseg/8_results_mslesseg/02_metrics"

# OPENMS_* -- model/checkpoint source (env.sh exports these unconditionally, so a
# ${x:-default} fallback never fires; override outright here, same gotcha as
# nnUNet_results in scripts/cluster/tamia_env.sh).
export OPENMS_DATASET_ROOT="$SCRATCH/open-ms"
export OPENMS_PREDICTIONS_ROOT="$OPENMS_DATASET_ROOT/8_results_open-ms/01_predictions"
export OPENMS_NNUNET_RAW="$OPENMS_DATASET_ROOT/2_nnUNet_open-ms/raw"
export OPENMS_NNUNET_PREPROCESSED="$OPENMS_DATASET_ROOT/2_nnUNet_open-ms/preprocessed"
export OPENMS_DATASET_JSON="$OPENMS_NNUNET_RAW/${OPENMS_DS_NAME:-Dataset070_OpenMS_FLAIR}/dataset.json"

# run_job overrides for tamia (whole-node H100, see CLAUDE.md "TamIA" section).
export RUN_JOB_ACCOUNT="aip-jcohen"
export RUN_JOB_GPU_TYPE="h100"

mkdir -p "$PREDICTIONS_ROOT" "$METRICS_ROOT"
