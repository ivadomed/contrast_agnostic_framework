#!/usr/bin/env bash
# Predict on ispy2's own test cases with the T2W ladder rung-2 model
# (baseline+K-means intensity clustering). Was missing 2026-09-06 (t2w ladder
# had not been predicted/evaluated yet on the own-eval side; only t1wce's rungs
# 2-4 existed).
# Usage: bash 05_45_predict_own_t2w_baseline_kmeans.sh <RUN_ID> [FOLD]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t2w.sh"
METHOD="baseline_kmeans"
TRAINER="nnUNetTrainerISPY2AugLabDefault"
CATEGORY="auglab"
DATASET_ID="${DATASET_ID_T2W}"
source "$(dirname "$0")/05_20_predict_common.sh" "$@"
