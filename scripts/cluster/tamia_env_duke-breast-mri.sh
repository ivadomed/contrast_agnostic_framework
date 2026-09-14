# Source AFTER datasets/duke-breast-mri/5_scripts_duke-breast-mri/00_utils/env.sh,
# to run the ispy2 -> duke-breast-mri cross-dataset predict/evaluate direction
# entirely on tamia scratch. Modeled directly on
# scripts/cluster/tamia_env_ambl_ispy2_target.sh (same shape: an eval-only
# dataset predicted on by ispy2-trained models) -- READ THAT FILE'S HEADER
# COMMENT FIRST, it documents a real /project file-count quota outage this
# project already hit once from writing too many cross-dataset prediction
# files there (~2150 files from ambl's 69-case suite alone maxed the shared
# quota). Duke has 291 cases (~4x ambl) across 12 runs x 3 folds x 1 item
# (duke only has one modality) = ~1050 predict-output dirs -- put everything
# on scratch from the start, do not repeat that mistake.
#
#   source datasets/duke-breast-mri/5_scripts_duke-breast-mri/00_utils/env.sh
#   source scripts/cluster/tamia_env_duke-breast-mri.sh
#
# Override every path outright (not with a ${VAR:-default} guard) -- see
# CLAUDE.md's TamIA section: env.sh's own ${VAR:-default} guards already fired
# by the time this file sources, so a guard here would be a no-op.

export SCRATCH="${SCRATCH:-/scratch/p/paulh}"   # tamia: extra /p/ nesting, unset in non-login shells
ISPY2_SCRATCH="$SCRATCH/ispy2"
DUKE_SCRATCH="$SCRATCH/duke-breast-mri"

# duke-breast-mri's OWN paths (test-set GT + prediction/metric OUTPUT) --
# scratch from the start (see header).
export nnUNet_raw="$DUKE_SCRATCH/2_nnUNet/raw"
export PREDICTIONS_ROOT="$DUKE_SCRATCH/8_results/01_predictions"
export METRICS_ROOT="$DUKE_SCRATCH/8_results/02_metrics"
mkdir -p "$PREDICTIONS_ROOT" "$METRICS_ROOT"

# ISPY2_* cross-dataset MODEL-SOURCE vars -- point at scratch-resident ispy2
# checkpoints/raw (same source ambl's cross-eval already reads from).
export ISPY2_DATASET_ROOT="$ISPY2_SCRATCH"
export ISPY2_NNUNET_RAW="$ISPY2_SCRATCH/2_nnUNet/raw"
export ISPY2_NNUNET_PREPROCESSED="$ISPY2_SCRATCH/2_nnUNet/preprocessed"
export ISPY2_PREDICTIONS_ROOT="$ISPY2_SCRATCH/8_results/01_predictions"
export ISPY2_DATASET_JSON="$ISPY2_NNUNET_RAW/Dataset100_ISPY2T1wce/dataset.json"

export RUN_JOB_ACCOUNT="aip-jcohen"
export RUN_JOB_GPU_TYPE="h100"
