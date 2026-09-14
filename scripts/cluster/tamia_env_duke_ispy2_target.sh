# Source AFTER datasets/duke-breast-mri/5_scripts_duke-breast-mri/00_utils/env.sh,
# to run the ispy2 -> duke-breast-mri cross-dataset predict/evaluate direction
# entirely on tamia scratch. Mirrors
# scripts/cluster/tamia_env_ambl_ispy2_target.sh's pattern exactly (same
# /project file-count quota lesson from that file's header applies here:
# ~291 cases x up to 12 runs x 3 folds x 1 item is a large file count, keep
# duke's own predict/metrics OUTPUT on scratch, not /project).
#
#   source datasets/duke-breast-mri/5_scripts_duke-breast-mri/00_utils/env.sh
#   source scripts/cluster/tamia_env_duke_ispy2_target.sh
#
# Override every path outright (not with a ${VAR:-default} guard) -- see
# CLAUDE.md's TamIA section: env.sh's own ${VAR:-default} guards already fired
# by the time this file sources, so a guard here would be a no-op.

export SCRATCH="${SCRATCH:-/scratch/p/paulh}"   # tamia: extra /p/ nesting, unset in non-login shells
ISPY2_SCRATCH="$SCRATCH/ispy2"
DUKE_SCRATCH="$SCRATCH/duke-breast-mri"

# duke-breast-mri's OWN paths (test-set GT + prediction/metric OUTPUT) -- on
# scratch from the start (unlike ambl, never staged to /project in the first
# place, so no migration needed here).
export nnUNet_raw="$DUKE_SCRATCH/2_nnUNet/raw"
export PREDICTIONS_ROOT="$DUKE_SCRATCH/8_results/01_predictions"
export METRICS_ROOT="$DUKE_SCRATCH/8_results/02_metrics"
mkdir -p "$PREDICTIONS_ROOT" "$METRICS_ROOT"

# ISPY2_* cross-dataset MODEL-SOURCE vars -- point at scratch-resident ispy2
# checkpoints/raw (same as ambl's identical override).
export ISPY2_DATASET_ROOT="$ISPY2_SCRATCH"
export ISPY2_NNUNET_RAW="$ISPY2_SCRATCH/2_nnUNet/raw"
export ISPY2_NNUNET_PREPROCESSED="$ISPY2_SCRATCH/2_nnUNet/preprocessed"
export ISPY2_PREDICTIONS_ROOT="$ISPY2_SCRATCH/8_results/01_predictions"
export ISPY2_DATASET_JSON="$ISPY2_NNUNET_RAW/Dataset100_ISPY2T1wce/dataset.json"

export RUN_JOB_ACCOUNT="aip-jcohen"
export RUN_JOB_GPU_TYPE="h100"
