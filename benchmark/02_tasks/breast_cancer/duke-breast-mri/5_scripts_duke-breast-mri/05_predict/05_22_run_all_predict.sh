#!/usr/bin/env bash
# Launch every ispy2-model prediction on duke-breast-mri (12 headline + 8 ladder-rung
# runs x 3 folds) for the items in ITEMS (default: the 2026-10-01 standard
# "t1wce_uniap precontrast_uniap", L-R + skin-anchored A-P crop). Spaced submissions;
# each wrapper blocks on its own fold jobs, so this returns when all are done.
#   nohup bash 05_22_run_all_predict.sh > <log> 2>&1 &
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ITEMS="${ITEMS:-t1wce_uniap precontrast_uniap}"
export RUN_JOB_CPUS_PER_GPU="${RUN_JOB_CPUS_PER_GPU:-8}" RUN_JOB_MEM_PER_GPU="${RUN_JOB_MEM_PER_GPU:-40G}"
export PREDICT_TIME_OVERRIDE="${PREDICT_TIME_OVERRIDE:-01:00:00}"
LOGD="${HERE}/../../8_results_duke-breast-mri/01_predictions"
for w in $(cd "${HERE}" && ls 05_0[2-9]_*.sh 05_1[0-9]_*.sh 05_2[01]_*.sh); do
    echo "[$(date '+%F %T')] launching ${w} (${ITEMS})"
    bash "${HERE}/${w}" "" all ${ITEMS} > "${LOGD}/_launch_${w%.sh}_uniap.log" 2>&1 &
    sleep "${SPACING_S:-20}"
done
wait
echo "[$(date '+%F %T')] all duke predictions finished"
