# Source AFTER datasets/lld-mmri-hcc/5_scripts_lld-mmri-hcc/00_utils/env.sh, to point
# LLD-MMRI-HCC's cross-dataset (atlas-liver-hcc-model) prediction at scratch-resident
# data on tamia.
#   source datasets/lld-mmri-hcc/5_scripts_lld-mmri-hcc/00_utils/env.sh
#   source scripts/cluster/tamia_env_lld-mmri-hcc_atlascross.sh
# Cluster differences are expressed as env overrides only — run_job.sh / predict_common.sh
# are never forked. Mirrors scripts/cluster/tamia_env_msd-spleen_chaoscross.sh's pattern.
#
# NOTE the atlas-liver-hcc scratch layout uses "2_nnUNet/" (no "_atlas-liver-hcc" name
# suffix), unlike Vulcan's "2_nnUNet_atlas-liver-hcc/" — see
# scripts/cluster/tamia_env_atlas-liver-hcc.sh (the script that originally staged it
# there) and /project/aip-jcohen/paulh/copy_atlas_liver_hcc_to_tamia.sh.

export SCRATCH="${SCRATCH:-/scratch/p/paulh}"   # tamia: extra /p/ nesting, unset in non-login shells

# LLD-MMRI-HCC's own paths (inputs for cross-mode predict + this dataset's own output tree).
# METRICS_ROOT intentionally left at common_env.sh's default (under $PROJECT, NOT
# scratch) -- same reasoning as msd-spleen/cirrmri-liver/kidney-t2w: predictions are
# large/many-file and go to scratch, but metrics CSVs are small/precious and land in
# the backed-up $PROJECT checkout. After the job, pull results back with
# scripts/cluster/fetch_tamia_results.sh (or the dataset-specific fetch script).
export nnUNet_raw="$SCRATCH/lld-mmri-hcc/2_nnUNet_lld-mmri-hcc/raw"
export PREDICTIONS_ROOT="$SCRATCH/lld-mmri-hcc/8_results_lld-mmri-hcc/01_predictions"

# ATLAS_* — model/checkpoint source (env.sh exports these unconditionally, so a
# ${x:-default} fallback never fires; override outright here, same gotcha as
# nnUNet_results in scripts/cluster/tamia_env.sh). atlas-liver-hcc's baseline
# checkpoints live only under TamIA's own $PROJECT (a training-time clobber-bug
# artifact, see project_atlas_liver_hcc_onboarding memory) -- reached via the
# symlink already created at
# $SCRATCH/atlas-liver-hcc/8_results/01_predictions/atlas_liver_hcc_model/t1w/nnUNet/
# atlas-liver-hcc_t1w_baseline_20260804_062605 (valid ON tamia; do not recreate it
# elsewhere or the auglab-category runs' real scratch-resident dirs get shadowed).
export ATLAS_DATASET_ROOT="$SCRATCH/atlas-liver-hcc"
export ATLAS_PREDICTIONS_ROOT="$ATLAS_DATASET_ROOT/8_results/01_predictions"
export ATLAS_NNUNET_RAW="$ATLAS_DATASET_ROOT/2_nnUNet/raw"
export ATLAS_NNUNET_PREPROCESSED="$ATLAS_DATASET_ROOT/2_nnUNet/preprocessed"
export ATLAS_DATASET_ID="80"
export ATLAS_DS_NAME="Dataset080_AtlasLiverHCC"
export ATLAS_DATASET_JSON="$ATLAS_NNUNET_RAW/$ATLAS_DS_NAME/dataset.json"
export ATLAS_TRAINING_CONTRAST="t1w"
export ATLAS_MODEL_TYPE="atlas_liver_hcc_model"

# run_job overrides for tamia (whole-node H100, see CLAUDE.md "TamIA" section).
export RUN_JOB_ACCOUNT="aip-jcohen"
export RUN_JOB_GPU_TYPE="h100"

mkdir -p "$PREDICTIONS_ROOT"
