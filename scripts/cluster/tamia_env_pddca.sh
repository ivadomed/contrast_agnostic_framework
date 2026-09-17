# Source AFTER datasets/pddca/5_scripts_pddca/00_utils/env.sh on tamia.
# pddca is TEST-ONLY: its own dirs hold the prepared CT test inputs, while the
# TF2_* vars must point at the toothfairy2 TRAINED model, which on tamia lives only
# under /scratch/p/paulh/toothfairy2 (CLAUDE.md: bulk data on tamia is scratch-only).
# Override outright — common_env's ${VAR:-default} guards have already fired.
export SCRATCH="${SCRATCH:-/scratch/p/paulh}"
# ⚠️ Underscore-prefixed, dataset-qualified names ON PURPOSE. This file is SOURCED
# into a caller's shell, so any short generic name it sets silently overwrites the
# caller's variable of the same name. Earlier versions used bare `HS` and `TF2`,
# which clobbered the script-directory variables of the same names in
# toothfairy2's 05_31 predict+evaluate job (hit on hanseg; same trap here) — every pddca predict wrapper then
# resolved to "/scratch/p/paulh/pddca/05_predict/..." instead of the repo path,
# and the ENTIRE cross-modality half of the run recorded zero tasks while the
# in-domain half succeeded. It failed loudly in a side log but the job carried on,
# which is exactly how a half-empty results table gets produced. Never introduce a
# short unprefixed variable in a sourced env file.
_PDDCA_SCRATCH="$SCRATCH/pddca"; _TF2_SCRATCH="$SCRATCH/toothfairy2"

export BIDS_ROOT="$_PDDCA_SCRATCH/1_BIDS/headneck-pddca"
export RAW_ROOT="$_PDDCA_SCRATCH/0_raw"
export nnUNet_raw="$_PDDCA_SCRATCH/2_nnUNet/raw"
export SPLITS_DIR="$_PDDCA_SCRATCH/4_splits"
export PREDICTIONS_ROOT="$_PDDCA_SCRATCH/8_results/01_predictions"
export METRICS_ROOT="$_PDDCA_SCRATCH/8_results/02_metrics"
export RESULTS_DIR="$_PDDCA_SCRATCH/8_results"

export TF2_PREDICTIONS_ROOT="$_TF2_SCRATCH/8_results/01_predictions"
export TF2_NNUNET_RAW="$_TF2_SCRATCH/2_nnUNet/raw"
export TF2_NNUNET_PREPROCESSED="$_TF2_SCRATCH/2_nnUNet/preprocessed"
export TF2_DATASET_JSON="$_TF2_SCRATCH/2_nnUNet/raw/Dataset110_ToothFairy2CBCT/dataset.json"

export RUN_JOB_ACCOUNT="aip-jcohen"
export RUN_JOB_GPU_TYPE="h100"
mkdir -p "$PREDICTIONS_ROOT" "$METRICS_ROOT" "$BIDS_ROOT" "$nnUNet_raw"
