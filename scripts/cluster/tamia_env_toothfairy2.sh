# Source AFTER datasets/toothfairy2/5_scripts_toothfairy2/00_utils/env.sh, to point
# toothfairy2's data + training at scratch-resident paths on tamia. Same pattern as
# tamia_env_ispy2.sh / tamia_env_ambl.sh.
#
#   source datasets/toothfairy2/5_scripts_toothfairy2/00_utils/env.sh
#   source scripts/cluster/tamia_env_toothfairy2.sh
#
# env.sh (via common_env.sh) exports every path with a ${VAR:-default} guard, and by
# the time THIS file runs those guards have already fired with the git-repo-relative
# default — so a ${VAR:-...} guard HERE would be a silent no-op (CLAUDE.md's TamIA
# gotcha). Override outright.
#
# ⚠️ BIDS_ROOT is overridden too, unlike ispy2's override file. toothfairy2's BIDS
# tree is ~480 volumes of resampled CBCT; on TamIA, $PROJECT is under a binding
# FILE-COUNT quota and CLAUDE.md's standing rule is that all bulk imaging data on
# this cluster lives on $SCRATCH only. Treat the scratch copy as re-creatable
# (purge-on-inactivity): 00_utils/00_00_extract_and_bidsify.sh rebuilds it from the
# release zip.

export SCRATCH="${SCRATCH:-/scratch/p/paulh}"   # tamia: extra /p/ nesting, unset in non-login shells
TF2_SCRATCH="$SCRATCH/toothfairy2"

export BIDS_ROOT="$TF2_SCRATCH/1_BIDS/maxillofacial-toothfairy2"
export RAW_ROOT="$TF2_SCRATCH/0_raw"
export nnUNet_raw="$TF2_SCRATCH/2_nnUNet/raw"
export nnUNet_preprocessed="$TF2_SCRATCH/2_nnUNet/preprocessed"
export nnUNet_results="$TF2_SCRATCH/8_results/01_predictions/toothfairy2_model/${TRAINING_CONTRAST:-cbct}/nnUNet"
export PREDICTIONS_ROOT="$TF2_SCRATCH/8_results/01_predictions"
export METRICS_ROOT="$TF2_SCRATCH/8_results/02_metrics"
export SPLITS_DIR="$TF2_SCRATCH/4_splits"
export RESULTS_DIR="$TF2_SCRATCH/8_results"

export RUN_JOB_ACCOUNT="aip-jcohen"
export RUN_JOB_GPU_TYPE="h100"

mkdir -p "$nnUNet_results" "$PREDICTIONS_ROOT" "$METRICS_ROOT" "$SPLITS_DIR" "$BIDS_ROOT"
