# Source AFTER datasets/hanseg/5_scripts_hanseg/00_utils/env.sh on tamia.
# hanseg is TEST-ONLY: its own dirs hold the prepared CT test inputs, while the
# TF2_* vars must point at the toothfairy2 TRAINED model, which on tamia lives only
# under /scratch/p/paulh/toothfairy2 (CLAUDE.md: bulk data on tamia is scratch-only).
# Override outright — common_env's ${VAR:-default} guards have already fired.
export SCRATCH="${SCRATCH:-/scratch/p/paulh}"
HS="$SCRATCH/hanseg"; TF2="$SCRATCH/toothfairy2"

export BIDS_ROOT="$HS/1_BIDS/headneck-hanseg"
export RAW_ROOT="$HS/0_raw"
export nnUNet_raw="$HS/2_nnUNet/raw"
export SPLITS_DIR="$HS/4_splits"
export PREDICTIONS_ROOT="$HS/8_results/01_predictions"
export METRICS_ROOT="$HS/8_results/02_metrics"
export RESULTS_DIR="$HS/8_results"

export TF2_PREDICTIONS_ROOT="$TF2/8_results/01_predictions"
export TF2_NNUNET_RAW="$TF2/2_nnUNet/raw"
export TF2_NNUNET_PREPROCESSED="$TF2/2_nnUNet/preprocessed"
export TF2_DATASET_JSON="$TF2/2_nnUNet/raw/Dataset110_ToothFairy2CBCT/dataset.json"

export RUN_JOB_ACCOUNT="aip-jcohen"
export RUN_JOB_GPU_TYPE="h100"
mkdir -p "$PREDICTIONS_ROOT" "$METRICS_ROOT" "$BIDS_ROOT" "$nnUNet_raw"
