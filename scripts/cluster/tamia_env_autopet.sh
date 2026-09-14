# Source AFTER datasets/autopet/5_scripts_autopet/00_utils/env.sh (or env_pet.sh), to
# point autopet's data + training at scratch-resident paths on tamia. Same pattern as
# tamia_env_toothfairy2.sh / tamia_env_ispy2.sh.
#
#   source datasets/autopet/5_scripts_autopet/00_utils/env.sh   # or env_pet.sh
#   source scripts/cluster/tamia_env_autopet.sh
#
# env.sh (via common_env.sh) exports every path with a ${VAR:-default} guard, and by the
# time THIS file runs those guards have already fired with the git-repo-relative default
# — a ${VAR:-...} guard HERE would be a silent no-op (CLAUDE.md's TamIA gotcha). Override
# outright.
#
# ⚠️ Everything lives on $SCRATCH, nothing in $PROJECT — same as every other dataset's
# tamia override (CLAUDE.md: "$PROJECT" is under a binding file-count quota on tamia, and
# whole-body PET/CT volumes are large besides). Treat the scratch copy as re-creatable
# (purge-on-inactivity) from the FDAT archive, not durable.

export SCRATCH="${SCRATCH:-/scratch/p/paulh}"   # tamia: extra /p/ nesting, unset in non-login shells
AUTOPET_SCRATCH="$SCRATCH/autopet"

export nnUNet_raw="$AUTOPET_SCRATCH/2_nnUNet/raw"
export nnUNet_preprocessed="$AUTOPET_SCRATCH/2_nnUNet/preprocessed"
export nnUNet_results="$AUTOPET_SCRATCH/8_results/01_predictions/${MODEL_TYPE:-autopet_model}/${TRAINING_CONTRAST:-ct}/nnUNet"
export PREDICTIONS_ROOT="$AUTOPET_SCRATCH/8_results/01_predictions"
export METRICS_ROOT="$AUTOPET_SCRATCH/8_results/02_metrics"
export SPLITS_DIR="$AUTOPET_SCRATCH/4_splits"
export RESULTS_DIR="$AUTOPET_SCRATCH/8_results"
# The raw FDAT archive itself — re-stage from Vulcan's /scratch/paulh/pet_task_staging/
# (or re-download from FDAT directly, no account needed) if this copy is purged.
export AUTOPET_ARCHIVE_ZIP="${AUTOPET_ARCHIVE_ZIP:-$AUTOPET_SCRATCH/psma-fdg-pet-ct-lesions_v2.zip}"
export AUTOPET_ARCHIVE_EXTRACT_DIR="${AUTOPET_ARCHIVE_EXTRACT_DIR:-$AUTOPET_SCRATCH/extracted}"

export RUN_JOB_ACCOUNT="aip-jcohen"
export RUN_JOB_GPU_TYPE="h100"
export RUN_JOB_CPUS_PER_GPU="12"
export RUN_JOB_MEM_PER_GPU="115G"   # GiB, NOT MB — see CLAUDE.md's TamIA memory gotcha

mkdir -p "$nnUNet_results" "$PREDICTIONS_ROOT" "$METRICS_ROOT" "$SPLITS_DIR"
