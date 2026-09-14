# Source AFTER datasets/ambl/5_scripts_ambl/00_utils/env.sh, to run the ispy2 ->
# ambl cross-dataset predict/evaluate direction entirely on tamia scratch.
#
# CORRECTED 2026-09-06 (was wrong on first write): this file originally left
# ambl's OWN nnUNet_raw/PREDICTIONS_ROOT/METRICS_ROOT at their git-repo-relative
# /project/... defaults, reasoning the malignant-only test set + its
# 8_results_ambl output were "small". That was true for the 276-file test set
# alone, but writing predictions for 8 runs x 3 folds x 2 items (~2150 files)
# hit `diskusage_report`'s /project (aip-jcohen) FILE-COUNT quota, which was
# already at 500K/500K (100%, group-wide, not just this user) -- predict jobs
# died mid-run with `OSError: [Errno 122] Disk quota exceeded`, producing
# SILENTLY PARTIAL prediction dirs (some folds 0/69 or a handful of cases) that
# looked like ordinary job output until file-count verification caught it. The
# partial tree was moved to /scratch/p/paulh/ambl/_toDelete/ (not deleted
# outright) and the malignant-only test set itself was relocated from /project
# to /scratch/p/paulh/ambl/2_nnUNet/raw/ to relieve pressure on the shared
# quota. Both ambl's own paths AND the ISPY2_* source vars now point at
# scratch -- unlike ambl's OLD training-role tamia_env_ambl.sh (still points
# at /scratch/p/paulh/ambl/... too, for the abandoned training role -- either
# file's override for ambl's OWN raw/results ends up at the same scratch host,
# but do not source that file here, it does not set the ISPY2_* vars this
# cross-dataset direction needs).
#
#   source datasets/ambl/5_scripts_ambl/00_utils/env.sh
#   source scripts/cluster/tamia_env_ambl_ispy2_target.sh
#
# Override every path outright (not with a ${VAR:-default} guard) -- see
# CLAUDE.md's TamIA section: env.sh's own ${VAR:-default} guards already fired
# by the time this file sources, so a guard here would be a no-op.

export SCRATCH="${SCRATCH:-/scratch/p/paulh}"   # tamia: extra /p/ nesting, unset in non-login shells
ISPY2_SCRATCH="$SCRATCH/ispy2"
AMBL_SCRATCH="$SCRATCH/ambl"

# ambl's OWN paths (test-set GT + prediction/metric OUTPUT) -- moved off
# /project (file-count quota) onto scratch.
export nnUNet_raw="$AMBL_SCRATCH/2_nnUNet/raw"
export PREDICTIONS_ROOT="$AMBL_SCRATCH/8_results/01_predictions"
export METRICS_ROOT="$AMBL_SCRATCH/8_results/02_metrics"
mkdir -p "$PREDICTIONS_ROOT" "$METRICS_ROOT"

# ISPY2_* cross-dataset MODEL-SOURCE vars (added to env.sh 2026-09-06) --
# point at scratch-resident ispy2 checkpoints/raw.
export ISPY2_DATASET_ROOT="$ISPY2_SCRATCH"
export ISPY2_NNUNET_RAW="$ISPY2_SCRATCH/2_nnUNet/raw"
export ISPY2_NNUNET_PREPROCESSED="$ISPY2_SCRATCH/2_nnUNet/preprocessed"
export ISPY2_PREDICTIONS_ROOT="$ISPY2_SCRATCH/8_results/01_predictions"
# dataset.json needed for label names only (background/tumour) -- read from
# scratch here for consistency with the other ISPY2_* overrides above.
export ISPY2_DATASET_JSON="$ISPY2_NNUNET_RAW/Dataset100_ISPY2T1wce/dataset.json"

export RUN_JOB_ACCOUNT="aip-jcohen"
export RUN_JOB_GPU_TYPE="h100"
