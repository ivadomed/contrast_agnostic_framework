# Source AFTER datasets/ambl/5_scripts_ambl/00_utils/env.sh (or env_t2w.sh), to point
# ambl's OWN training/prediction at scratch-resident data on tamia. Mirrors
# scripts/cluster/tamia_env_chaos.sh's pattern (own-training override, TWO training
# contrasts via TRAINING_CONTRAST -- t1wce default, t2w via env_t2w.sh) rather than
# tamia_env_atlas-liver-hcc.sh's single-contrast pattern, since ambl trains t1wce+t2w
# like chaos/brats2024-glioma/on-harmony/open-ms.
#   source datasets/ambl/5_scripts_ambl/00_utils/env.sh            # or env_t2w.sh
#   source scripts/cluster/tamia_env_ambl.sh
#
# env.sh/env_t2w.sh export nnUNet_results/PREDICTIONS_ROOT/METRICS_ROOT/SPLITS_DIR
# via common_env.sh's ${VAR:-default} guards -- by the time THIS file sources (second),
# those guards have already fired with the git-repo-relative default, so a ${VAR:-...}
# guard HERE would be a no-op (see CLAUDE.md's TamIA section gotcha, hit once already
# on brats2024-glioma/atlas-liver-hcc/chaos). Override every path outright.

export SCRATCH="${SCRATCH:-/scratch/p/paulh}"   # tamia: extra /p/ nesting, unset in non-login shells
AMBL_SCRATCH="$SCRATCH/ambl"

export nnUNet_raw="$AMBL_SCRATCH/2_nnUNet/raw"
export nnUNet_preprocessed="$AMBL_SCRATCH/2_nnUNet/preprocessed"
export nnUNet_results="$AMBL_SCRATCH/8_results/01_predictions/ambl_model/${TRAINING_CONTRAST:-t1wce}/nnUNet"
export PREDICTIONS_ROOT="$AMBL_SCRATCH/8_results/01_predictions"
export METRICS_ROOT="$AMBL_SCRATCH/8_results/02_metrics"
export SPLITS_DIR="$AMBL_SCRATCH/4_splits"

export RUN_JOB_ACCOUNT="aip-jcohen"
export RUN_JOB_GPU_TYPE="h100"

mkdir -p "$nnUNet_results" "$PREDICTIONS_ROOT" "$METRICS_ROOT"
