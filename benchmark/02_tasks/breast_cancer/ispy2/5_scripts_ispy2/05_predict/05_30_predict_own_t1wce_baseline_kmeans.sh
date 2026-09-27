#!/usr/bin/env bash
# Predict on ispy2's own test cases with the t1wce ladder rung-2 model
# (baseline+K-means intensity clustering).
# Usage: bash 05_30_predict_own_t1wce_baseline_kmeans.sh <RUN_ID> [FOLD]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="baseline_kmeans"
TRAINER="nnUNetTrainerISPY2AugLabDefault"
CATEGORY="auglab"
DATASET_ID="${DATASET_ID_T1WCE}"
source "$(dirname "$0")/05_20_predict_common.sh" "$@"
