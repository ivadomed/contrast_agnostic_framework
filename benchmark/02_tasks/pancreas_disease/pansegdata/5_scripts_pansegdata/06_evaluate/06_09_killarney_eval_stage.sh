#!/usr/bin/env bash
# Killarney eval STAGE (a CPU job queued by 05_29 behind the predict jobs): evaluates every pinned roster run of both contrasts INLINE (already inside a compute
# allocation, so no nested run_job) via the shared drivers, then checks that the metrics are complete. Metrics land in 8_results_pansegdata/02_metrics/ (repo
# project space on Killarney). The manual tail afterwards: 06_05_write_configs.sh -> 06_07_run_all_aggregation.sh (here or after rsync to Vulcan).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
unset RUN_JOB_DEPENDENCY
source "${HERE}/../00_utils/env.sh"
export EVAL_INLINE=1
OUTD="${RESULTS_DIR}/post_training"; mkdir -p "${OUTD}"
bash "${HERE}/06_06_run_all_eval.sh" all; rc=$?
n=$(find "${METRICS_ROOT}/pansegdata_model" -name eval_all.csv 2>/dev/null | wc -l)
echo "[eval-stage] rc=${rc}; eval_all.csv files: ${n} (expect 22 runs x 3 folds = 66)"
if [ "${rc}" = 0 ] && [ "${n}" -ge 66 ]; then echo "EVAL COMPLETE $(date)" > "${OUTD}/DONE_eval.txt"; else echo "EVAL INCOMPLETE rc=${rc} n=${n} $(date)" > "${OUTD}/DONE_eval.txt"; fi
cat "${OUTD}/DONE_eval.txt"
