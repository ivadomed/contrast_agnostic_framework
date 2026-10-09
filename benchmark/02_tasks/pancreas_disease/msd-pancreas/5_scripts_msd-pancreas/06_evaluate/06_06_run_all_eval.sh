#!/usr/bin/env bash
# Evaluate the whole pansegdata roster on msd-pancreas's items (headline -> <contrast>/<item>/, ladder rungs -> <contrast>/ablations/<item>/, automatic) via the shared
# run_all_evaluate_cross_common.sh. env: ROSTER_ONLY, ROSTER_SKIP_MISSING=1, CKPT_TAG, EVAL_INLINE=1 (inside a job).   bash 06_06_run_all_eval.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
export SOURCE_PREFIX="PANSEG" SOURCE_CONTRASTS="${SOURCE_CONTRASTS:-t1wce t2w}" EVAL_SCRIPT="${HERE}/06_01_evaluate_run.sh"
source "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_03_evaluate/run_all_evaluate_cross_common.sh"
