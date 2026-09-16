# Source AFTER datasets/totalseg-pelvic/5_scripts_totalseg-pelvic/00_utils/env.sh (or
# env_mri.sh), to point totalseg-pelvic's OWN training/prediction at scratch-resident
# data on tamia. Mirrors scripts/cluster/tamia_env_chaos.sh's pattern (own-training
# override) -- totalseg-pelvic is CT+MRI like chaos, but UNPAIRED with a per-modality
# splits dir (4_splits_totalseg-pelvic/{ct,mri}/), unlike chaos's single shared split.
#   source datasets/totalseg-pelvic/5_scripts_totalseg-pelvic/00_utils/env.sh       # or env_mri.sh
#   source scripts/cluster/tamia_env_totalseg-pelvic.sh

export SCRATCH="${SCRATCH:-/scratch/p/paulh}"   # tamia: extra /p/ nesting, unset in non-login shells
TOTALSEG_PELVIC_SCRATCH="$SCRATCH/totalseg-pelvic"

export nnUNet_raw="$TOTALSEG_PELVIC_SCRATCH/2_nnUNet_totalseg-pelvic/raw"
export nnUNet_preprocessed="$TOTALSEG_PELVIC_SCRATCH/2_nnUNet_totalseg-pelvic/preprocessed"
# nnUNet_results/PREDICTIONS_ROOT: env.sh exports these UNCONDITIONALLY (same gotcha as
# chaos/brats' tamia_env files) -- override outright here, sourced last.
export nnUNet_results="$TOTALSEG_PELVIC_SCRATCH/8_results_totalseg-pelvic/01_predictions/totalseg_pelvic_model/${TRAINING_CONTRAST:-ct}/nnUNet"
export PREDICTIONS_ROOT="$TOTALSEG_PELVIC_SCRATCH/8_results_totalseg-pelvic/01_predictions"
export METRICS_ROOT="$TOTALSEG_PELVIC_SCRATCH/8_results_totalseg-pelvic/02_metrics"

# CT/MRI are unpaired (separate split dir per modality, see env.sh's own override) --
# reproduce that same per-contrast split path here rather than one shared SPLITS_DIR.
export SPLITS_DIR="$TOTALSEG_PELVIC_SCRATCH/4_splits_totalseg-pelvic/${TRAINING_CONTRAST:-ct}"

export CHECKPOINTS_DIR="$TOTALSEG_PELVIC_SCRATCH/6_checkpoints_totalseg-pelvic"
export RESULTS_DIR="$TOTALSEG_PELVIC_SCRATCH/8_results_totalseg-pelvic"

export RUN_JOB_ACCOUNT="aip-jcohen"
export RUN_JOB_GPU_TYPE="h100"

mkdir -p "$nnUNet_results" "$PREDICTIONS_ROOT" "$METRICS_ROOT" "$CHECKPOINTS_DIR" "$RESULTS_DIR/_logs"
