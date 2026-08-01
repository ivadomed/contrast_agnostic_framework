# Source AFTER datasets/ms3seg/5_scripts_ms3seg/00_utils/env.sh (or env_t1w.sh), to
# point ms3seg's cross-dataset (open-ms-model) prediction at scratch-resident data on
# tamia. Mirrors tamia_env_mslesseg_openmscross.sh exactly.
export SCRATCH="${SCRATCH:-/scratch/p/paulh}"

# NOTE: PREDICTIONS_ROOT (and METRICS_ROOT where set) point at TamIA $SCRATCH
# because TamIA's $PROJECT has a hard file-COUNT quota and a prediction tree is
# tens of thousands of small files. $SCRATCH is PURGE-ON-INACTIVITY, so it is
# NOT their permanent home -- after any TamIA predict/evaluate job, pull the
# results back into the Vulcan repo with:
#     bash scripts/cluster/fetch_tamia_results.sh <dataset>
# Skipping this left msd-spleen, cirrmri-liver, kidney-t2w, ms3seg and
# brats-ssa2024 with ZERO predictions on Vulcan (found 2026-08-01) -- metrics
# had been copied back but the predictions, which are what you actually need to
# inspect WHY a method scored what it did, had not.
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
