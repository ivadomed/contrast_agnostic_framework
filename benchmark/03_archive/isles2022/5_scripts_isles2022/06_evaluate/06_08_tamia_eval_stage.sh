#!/usr/bin/env bash
# TamIA eval STAGE (a CPU job queued by 05_28 after the predict packs): evaluates every pinned roster run of both contrasts INLINE (we are already in a
# compute allocation, so no nested run_job) via the shared drivers, then checks the metrics are complete. Metrics land in
# $SCRATCH/isles2022/8_results/02_metrics/...; the manual tail (on Vulcan, the repo's filesystem home) is:
#   bash scripts/cluster/fetch_tamia_results.sh ...  (or rsync the 02_metrics tree)  ->  06_05_write_configs.sh  ->  06_07_run_all_aggregation.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../../../../.." && pwd)"
export SCRATCH="${SCRATCH:-/scratch/p/paulh}"
source "${HERE}/../00_utils/env.sh"; source "${ROOT}/scripts/cluster/tamia_env_isles2022.sh"
export EVAL_INLINE=1
bash "${HERE}/06_06_run_all_eval.sh" all; rc=$?
n=$(find "${METRICS_ROOT}/isles2022_model" -name eval_all.csv 2>/dev/null | wc -l)
echo "[eval-stage] rc=${rc}; eval_all.csv files: ${n} (expect 22 runs x 3 folds = 66)"
if [ "${rc}" = 0 ] && [ "${n}" -ge 66 ]; then echo "EVAL COMPLETE $(date)" > "${SCRATCH}/isles2022/DONE_eval.txt"; else echo "EVAL INCOMPLETE rc=${rc} n=${n} $(date)" > "${SCRATCH}/isles2022/DONE_eval.txt"; exit 1; fi
