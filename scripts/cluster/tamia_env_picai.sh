# Source AFTER datasets/picai-prostate/5_scripts_picai-prostate/00_utils/env.sh (or
# env_adc.sh), to point picai-prostate's OWN training/prediction at scratch-resident data
# on tamia. Mirrors scripts/cluster/tamia_env_openms.sh's pattern (own-training override).
#   source datasets/picai-prostate/5_scripts_picai-prostate/00_utils/env_adc.sh
#   source scripts/cluster/tamia_env_picai.sh
#
# tamia's /project is at 492K/500K FILES (file count, not space, is the binding quota there),
# so every bulk path below must live on $SCRATCH. Only the repo + venv stay in /project.

export SCRATCH="${SCRATCH:-/scratch/p/paulh}"   # tamia: extra /p/ nesting, unset in non-login shells
PICAI_SCRATCH="$SCRATCH/picai-prostate"

# Raw + BIDS: picai-prostate downloads and BIDSifies ~27 GB of MHA on this cluster
# (tamia's login node is the only node with internet), so these are scratch-resident too —
# unlike open-ms/chaos, whose raw data was staged from vulcan.
export RAW_ROOT="$PICAI_SCRATCH/0_raw_picai-prostate"
export BIDS_ROOT="$PICAI_SCRATCH/1_BIDS_picai-prostate/picai-prostate-bpmri"

export nnUNet_raw="$PICAI_SCRATCH/2_nnUNet_picai-prostate/raw"
export nnUNet_preprocessed="$PICAI_SCRATCH/2_nnUNet_picai-prostate/preprocessed"
# nnUNet_results/PREDICTIONS_ROOT/METRICS_ROOT: env.sh exports these UNCONDITIONALLY
# (same gotcha as tamia_env_chaos.sh / tamia_env_openms.sh) — a ${x:-default} guard would
# silently no-op here. Override outright; this file is sourced last.
export nnUNet_results="$PICAI_SCRATCH/8_results_picai-prostate/01_predictions/picai_prostate_model/${TRAINING_CONTRAST:-t2w}/nnUNet"
export PREDICTIONS_ROOT="$PICAI_SCRATCH/8_results_picai-prostate/01_predictions"
export METRICS_ROOT="$PICAI_SCRATCH/8_results_picai-prostate/02_metrics"
export RESULTS_DIR="$PICAI_SCRATCH/8_results_picai-prostate"
# SPLITS_DIR stays in the repo (4_splits_picai-prostate is tiny JSON, and it is the
# scientific record of the partition) — do NOT move it to scratch, which is purgeable.

export RUN_JOB_ACCOUNT="aip-jcohen"
export RUN_JOB_GPU_TYPE="h100"

mkdir -p "$RAW_ROOT" "$nnUNet_raw" "$nnUNet_preprocessed" \
         "$nnUNet_results" "$PREDICTIONS_ROOT" "$METRICS_ROOT"
