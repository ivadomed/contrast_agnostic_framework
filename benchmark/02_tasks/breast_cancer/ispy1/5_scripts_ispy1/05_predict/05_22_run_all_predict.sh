#!/usr/bin/env bash
# Launch every ispy2-model prediction on ispy1 (12 headline + 8 ladder-rung runs,
# x 3 folds = 60 one-GPU run_job submissions). Submissions are spaced out
# (SPACING_S, default 20s) per cluster etiquette; each wrapper blocks on its own
# 3 fold jobs (--wait), so this script returns when all predictions are done.
#   nohup bash 05_22_run_all_predict.sh > <log> 2>&1 &
# Optional: pass wrapper basenames (e.g. 05_02_*.sh) to run a subset.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ $# -gt 0 ]; then WRAPPERS=("$@"); else
    mapfile -t WRAPPERS < <(cd "${HERE}" && ls 05_0[2-9]_*.sh 05_1[0-9]_*.sh 05_2[01]_*.sh)
fi
for w in "${WRAPPERS[@]}"; do
    echo "[$(date '+%F %T')] launching ${w}"
    bash "${HERE}/${w}" > "${HERE}/../../8_results_ispy1/01_predictions/_launch_${w%.sh}.log" 2>&1 &
    sleep "${SPACING_S:-20}"
done
wait
echo "[$(date '+%F %T')] all ispy1 predictions finished"
