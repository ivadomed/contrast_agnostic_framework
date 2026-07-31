# Source AFTER the dataset's env.sh to point nnUNet at the scratch-resident data.
#   source datasets/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh
#   source scripts/cluster/tamia_env.sh
# Cluster differences are expressed as env overrides only - run_job.sh is never forked.

export SCRATCH="${SCRATCH:-/scratch/p/paulh}"   # tamia: extra /p/ nesting, unset in non-login shells
BRATS_SCRATCH="$SCRATCH/brats2024-glioma"

export nnUNet_raw="$BRATS_SCRATCH/2_nnUNet/raw"
export nnUNet_preprocessed="$BRATS_SCRATCH/2_nnUNet/preprocessed"
# env.sh exports nnUNet_results UNCONDITIONALLY, so a ${x:-default} fallback never
# fires here. Override outright, otherwise checkpoints land in /project, whose FILE
# quota (483K/500K group-wide) is the binding constraint on this cluster.
export nnUNet_results="$BRATS_SCRATCH/8_results/nnUNet"
export SPLITS_DIR="$BRATS_SCRATCH/4_splits"
# PREDICTIONS_ROOT: same file-count-quota reasoning as nnUNet_results above -- predict_common.sh
# derives its OWN nnUNet_results (checkpoint lookup + prediction output) from this var,
# unconditionally overriding whatever env.sh/this file set nnUNet_results to above. Without
# this override predictions/checkpoint-lookup silently fall back to the /project path
# (common_env.sh's default), missing this scratch-resident data entirely. Was a real gap --
# added 2026-07-30 when the brats2024-glioma test-set expansion needed to predict on tamia.
export PREDICTIONS_ROOT="$BRATS_SCRATCH/8_results/01_predictions"

# run_job overrides for tamia
export RUN_JOB_ACCOUNT="aip-jcohen"
export RUN_JOB_GPU_TYPE="h100"

mkdir -p "$nnUNet_results"
mkdir -p "$PREDICTIONS_ROOT"
