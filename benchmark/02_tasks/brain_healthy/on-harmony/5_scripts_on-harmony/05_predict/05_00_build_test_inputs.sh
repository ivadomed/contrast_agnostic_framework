#!/usr/bin/env bash
# Materialize the shared cross-contrast test set as nnU-Net predict inputs. Run once
# (idempotent) before any 05_XX_predict_*.sh wrapper. Cheap (image I/O only, no GPU).
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

export BIDS_ROOT
export TEST_CASES="${PROJECT_ROOT}/benchmark/02_tasks/brain_healthy/on-harmony/4_splits_on-harmony/test_cases.json"
export TESTSET="${PREDICTIONS_ROOT}/${MODEL_TYPE}/_test_set"
export nnUNet_raw

.venv/bin/python "$(dirname "$0")/05_00_build_test_inputs.py"
