# Source AFTER datasets/cirrmri-liver/5_scripts_cirrmri-liver/00_utils/env.sh, to
# point CIRRMRI-LIVER's cross-dataset (chaos-model) prediction at scratch-resident
# data on tamia.
#   source datasets/cirrmri-liver/5_scripts_cirrmri-liver/00_utils/env.sh
#   source scripts/cluster/tamia_env_cirrmri-liver_chaoscross.sh
# Cluster differences are expressed as env overrides only — run_job.sh / predict_common.sh
# are never forked. Mirrors scripts/cluster/tamia_env_msd-spleen_chaoscross.sh's pattern.

export SCRATCH="${SCRATCH:-/scratch/p/paulh}"   # tamia: extra /p/ nesting, unset in non-login shells

# CIRRMRI-LIVER's own paths (inputs for cross-mode predict + this dataset's own output tree).
# METRICS_ROOT is intentionally left at common_env.sh's default (under $PROJECT,
# NOT scratch): TamIA's $PROJECT has a hard file-COUNT quota, and a prediction
# tree is tens of thousands of small files, so predictions must be WRITTEN to
# scratch here. But scratch is PURGE-ON-INACTIVITY, so scratch is not their
# permanent home -- after the job, pull them back to the Vulcan repo with:
#     bash scripts/cluster/fetch_tamia_results.sh <dataset>
# (Do not skip this. Predictions are the primary artifact for inspecting WHY a
# method scored what it scored; every Vulcan-run dataset keeps its own under
# 8_results_*/01_predictions. An earlier version of this comment called them
# "large/regenerable", which led to msd-spleen, cirrmri-liver and kidney-t2w
# all sitting with ZERO predictions on Vulcan -- found 2026-08-01.)
export nnUNet_raw="$SCRATCH/cirrmri-liver/2_nnUNet_cirrmri-liver/raw"
export PREDICTIONS_ROOT="$SCRATCH/cirrmri-liver/8_results_cirrmri-liver/01_predictions"

# CHAOS_* — model/checkpoint source (env.sh exports these unconditionally, so a
# ${x:-default} fallback never fires; override outright here, same gotcha as
# nnUNet_results in scripts/cluster/tamia_env.sh).
export CHAOS_DATASET_ROOT="$SCRATCH/chaos"
export CHAOS_PREDICTIONS_ROOT="$CHAOS_DATASET_ROOT/8_results_chaos/01_predictions"
export CHAOS_NNUNET_RAW="$CHAOS_DATASET_ROOT/2_nnUNet_chaos/raw"
export CHAOS_NNUNET_PREPROCESSED="$CHAOS_DATASET_ROOT/2_nnUNet_chaos/preprocessed"

# run_job overrides for tamia (whole-node H100, see CLAUDE.md "TamIA" section).
export RUN_JOB_ACCOUNT="aip-jcohen"
export RUN_JOB_GPU_TYPE="h100"

mkdir -p "$PREDICTIONS_ROOT"
