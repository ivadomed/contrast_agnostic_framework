# Source AFTER datasets/ispy2/5_scripts_ispy2/00_utils/env.sh (or env_t2w.sh), to
# point ispy2's OWN training/prediction at scratch-resident data on tamia. Mirrors
# scripts/cluster/tamia_env_ambl.sh's pattern (own-training override, TWO training
# contrasts via TRAINING_CONTRAST -- t1wce default, t2w via env_t2w.sh).
#
# REWRITTEN 2026-09-04 (pivot): this file previously supported ispy2's OLD role
# (cross-eval-only, evaluated against ambl-trained checkpoints) -- that layout is
# now stale. ispy2 is the PRIMARY TRAINING dataset; ambl is the external test set
# (the reverse direction). The AMBL_* cross-dataset vars are dropped here since
# nothing in the training path reads them; a future predict/eval phase (ispy2
# models -> ambl test set) will need its own override, symmetric to how ambl's
# old cross-eval scripts pointed at AMBL_* here.
#
#   source datasets/ispy2/5_scripts_ispy2/00_utils/env.sh            # or env_t2w.sh
#   source scripts/cluster/tamia_env_ispy2.sh
#
# env.sh/env_t2w.sh export nnUNet_results/PREDICTIONS_ROOT/METRICS_ROOT/SPLITS_DIR
# via common_env.sh's ${VAR:-default} guards -- by the time THIS file sources (second),
# those guards have already fired with the git-repo-relative default, so a ${VAR:-...}
# guard HERE would be a no-op (CLAUDE.md's TamIA section gotcha). Override every path
# outright.

export SCRATCH="${SCRATCH:-/scratch/p/paulh}"   # tamia: extra /p/ nesting, unset in non-login shells
ISPY2_SCRATCH="$SCRATCH/ispy2"

export nnUNet_raw="$ISPY2_SCRATCH/2_nnUNet/raw"
export nnUNet_preprocessed="$ISPY2_SCRATCH/2_nnUNet/preprocessed"
export nnUNet_results="$ISPY2_SCRATCH/8_results/01_predictions/ispy2_model/${TRAINING_CONTRAST:-t1wce}/nnUNet"
export PREDICTIONS_ROOT="$ISPY2_SCRATCH/8_results/01_predictions"
export METRICS_ROOT="$ISPY2_SCRATCH/8_results/02_metrics"
export SPLITS_DIR="$ISPY2_SCRATCH/4_splits"

export RUN_JOB_ACCOUNT="aip-jcohen"
export RUN_JOB_GPU_TYPE="h100"

mkdir -p "$nnUNet_results" "$PREDICTIONS_ROOT" "$METRICS_ROOT"
