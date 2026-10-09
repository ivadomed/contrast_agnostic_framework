#!/usr/bin/env bash
# Evaluate ONE run of the pansegdata roster on totalsegmri-pancreas's items (every fold x item), via the shared 00_03_evaluate/evaluate_companion_run_common.sh.
#   bash 06_01_evaluate_run.sh <RUN_ID> <CATEGORY:nnUNet|auglab> <SOURCE_TRAINING_CONTRAST:t1wce|t2w> [FOLD=all]     env: LADDER=1 (rungs), CKPT_TAG, METRICS_SUBDIR n/a
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
SOURCE_PREFIX="PANSEG"; EVAL_ITEMS="t1gre t2like"; EVAL_LABELS="pancreas"; EVAL_JOB_PREFIX="totalsegmri-pancreas_eval"
source "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_03_evaluate/evaluate_companion_run_common.sh" "$@"
