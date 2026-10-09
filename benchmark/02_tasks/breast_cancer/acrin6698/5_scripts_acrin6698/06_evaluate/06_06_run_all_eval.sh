#!/usr/bin/env bash
# Evaluate every acrin6698 prediction run: reads METHOD/CATEGORY/contrast/RUN_ID
# straight from the 05_predict wrappers (single source of truth), headline
# wrappers 05_02-05_13 -> <contrast>/<item>/, ladder rungs 05_14-05_21 ->
# <contrast>/ablations/<item>/ (LADDER=1). Spaced submissions, all in parallel.
#   nohup bash 06_06_run_all_eval.sh > <log> 2>&1 &
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRED="${HERE}/../05_predict"
for w in "${PRED}"/05_{0[2-9],1[0-9],2[01]}_*.sh; do
    n="$(basename "$w" | cut -d_ -f2)"
    cat="$(grep -oP '^CATEGORY="\K[^"]+' "$w")"
    tc="$(grep -oP 'ISPY2_TRAINING_CONTRAST="\K[^"]+' "$w")"
    rid="$(grep -oP 'RUN_ID="\$\{1:-\K[^}]+' "$w")"
    ladder=0; [ "$((10#$n))" -ge 14 ] && ladder=1
    echo "[$(date '+%F %T')] eval ${rid} (${cat}, ${tc}, ladder=${ladder})"
    LADDER=${ladder} bash "${HERE}/06_01_evaluate_run.sh" "${rid}" "${cat}" "${tc}" &
    sleep "${SPACING_S:-5}"
done
wait
echo "[$(date '+%F %T')] all acrin6698 evaluations finished"
