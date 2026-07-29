# Source AFTER datasets/ms3seg/5_scripts_ms3seg/00_utils/env.sh (or env_t1w.sh), to
# point ms3seg's cross-dataset (open-ms-model) prediction at scratch-resident data on
# tamia. Mirrors tamia_env_mslesseg_openmscross.sh exactly.
export SCRATCH="${SCRATCH:-/scratch/p/paulh}"

export PREDICTIONS_ROOT="$SCRATCH/ms3seg/8_results_ms3seg/01_predictions"
export METRICS_ROOT="$SCRATCH/ms3seg/8_results_ms3seg/02_metrics"

export OPENMS_DATASET_ROOT="$SCRATCH/open-ms"
export OPENMS_PREDICTIONS_ROOT="$OPENMS_DATASET_ROOT/8_results_open-ms/01_predictions"
export OPENMS_NNUNET_RAW="$OPENMS_DATASET_ROOT/2_nnUNet_open-ms/raw"
export OPENMS_NNUNET_PREPROCESSED="$OPENMS_DATASET_ROOT/2_nnUNet_open-ms/preprocessed"
# OPENMS_DATASET_ID/_DS_NAME/_TRAINING_CONTRAST: leave at whatever env.sh/env_t1w.sh
# already set -- do NOT override here.

export RUN_JOB_ACCOUNT="aip-jcohen"
export RUN_JOB_GPU_TYPE="h100"

mkdir -p "$PREDICTIONS_ROOT" "$METRICS_ROOT"
