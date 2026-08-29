# Source AFTER datasets/liverhccseg/5_scripts_liverhccseg/00_utils/env.sh, to point
# LiverHccSeg's cross-dataset (atlas-liver-hcc-model) prediction at scratch-resident
# data on tamia.
#   source datasets/liverhccseg/5_scripts_liverhccseg/00_utils/env.sh
#   source scripts/cluster/tamia_env_liverhccseg_atlascross.sh
# Cluster differences are expressed as env overrides only — run_job.sh / predict_common.sh
# are never forked. Mirrors scripts/cluster/tamia_env_lld-mmri-hcc_atlascross.sh's pattern
# exactly (see that file's note on the "2_nnUNet/" vs "2_nnUNet_atlas-liver-hcc/" naming
# quirk and the baseline-checkpoint symlink).

export SCRATCH="${SCRATCH:-/scratch/p/paulh}"   # tamia: extra /p/ nesting, unset in non-login shells

# LiverHccSeg's own paths (inputs for cross-mode predict + this dataset's own output tree).
# METRICS_ROOT intentionally left at common_env.sh's default (under $PROJECT, NOT
# scratch) -- small precious CSVs; predictions go to scratch (large/many files).
export nnUNet_raw="$SCRATCH/liverhccseg/2_nnUNet_liverhccseg/raw"
export PREDICTIONS_ROOT="$SCRATCH/liverhccseg/8_results_liverhccseg/01_predictions"

# ATLAS_* — model/checkpoint source (env.sh exports these unconditionally, so a
# ${x:-default} fallback never fires; override outright here).
export ATLAS_DATASET_ROOT="$SCRATCH/atlas-liver-hcc"
export ATLAS_PREDICTIONS_ROOT="$ATLAS_DATASET_ROOT/8_results/01_predictions"
export ATLAS_NNUNET_RAW="$ATLAS_DATASET_ROOT/2_nnUNet/raw"
export ATLAS_NNUNET_PREPROCESSED="$ATLAS_DATASET_ROOT/2_nnUNet/preprocessed"
export ATLAS_DATASET_ID="80"
export ATLAS_DS_NAME="Dataset080_AtlasLiverHCC"
export ATLAS_DATASET_JSON="$ATLAS_NNUNET_RAW/$ATLAS_DS_NAME/dataset.json"
export ATLAS_TRAINING_CONTRAST="t1w"
export ATLAS_MODEL_TYPE="atlas_liver_hcc_model"

export RUN_JOB_ACCOUNT="aip-jcohen"
export RUN_JOB_GPU_TYPE="h100"

mkdir -p "$PREDICTIONS_ROOT"
