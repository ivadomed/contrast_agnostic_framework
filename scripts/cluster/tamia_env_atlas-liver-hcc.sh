# Source AFTER datasets/atlas-liver-hcc/5_scripts_atlas-liver-hcc/00_utils/env.sh, to
# point atlas-liver-hcc's OWN training/prediction at scratch-resident data on tamia.
# Mirrors scripts/cluster/tamia_env_openms.sh's pattern (own-training override, single
# training contrast).
#   source datasets/atlas-liver-hcc/5_scripts_atlas-liver-hcc/00_utils/env.sh
#   source scripts/cluster/tamia_env_atlas-liver-hcc.sh

export SCRATCH="${SCRATCH:-/scratch/p/paulh}"   # tamia: extra /p/ nesting, unset in non-login shells
ATLAS_SCRATCH="$SCRATCH/atlas-liver-hcc"

export nnUNet_raw="$ATLAS_SCRATCH/2_nnUNet/raw"
export nnUNet_preprocessed="$ATLAS_SCRATCH/2_nnUNet/preprocessed"
# env.sh exports nnUNet_results/PREDICTIONS_ROOT/METRICS_ROOT UNCONDITIONALLY (same
# gotcha as chaos/open-ms's tamia_env overrides) -- override outright here, sourced last.
export nnUNet_results="$ATLAS_SCRATCH/8_results/01_predictions/atlas_liver_hcc_model/t1w/nnUNet"
export PREDICTIONS_ROOT="$ATLAS_SCRATCH/8_results/01_predictions"
export METRICS_ROOT="$ATLAS_SCRATCH/8_results/02_metrics"
export SPLITS_DIR="$ATLAS_SCRATCH/4_splits"

export RUN_JOB_ACCOUNT="aip-jcohen"
export RUN_JOB_GPU_TYPE="h100"

mkdir -p "$nnUNet_results" "$PREDICTIONS_ROOT" "$METRICS_ROOT"
