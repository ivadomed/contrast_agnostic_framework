#!/usr/bin/env bash
# Evaluate every duke prediction run for each item in ITEMS (default
# "t1wce_uniap precontrast_uniap"): reads METHOD/CATEGORY/contrast/RUN_ID from the
# 05_predict wrappers; headline 05_02-05_13 -> <contrast>/<item>/, ladder rungs
# 05_14-05_21 -> <contrast>/ablations/<item>/ (same layout as the *_uni items).
#   nohup bash 06_06_run_all_eval.sh > <log> 2>&1 &
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRED="${HERE}/../05_predict"
ITEMS="${ITEMS:-t1wce_uniap precontrast_uniap}"
for w in "${PRED}"/05_{0[2-9],1[0-9],2[01]}_*.sh; do
    n="$(basename "$w" | cut -d_ -f2)"
    cat="$(grep -oP '^CATEGORY="\K[^"]+' "$w")"
    tc="$(grep -oP 'ISPY2_TRAINING_CONTRAST="\K[^"]+' "$w")"
    rid="$(grep -oP 'RUN_ID="\$\{1:-\K[^}]+' "$w")"
    for item in ${ITEMS}; do
        sub="${item}"; [ "$((10#$n))" -ge 14 ] && sub="ablations/${item}"
        echo "[$(date '+%F %T')] eval ${rid} (${cat}, ${tc}) -> ${sub}"
        DUKE_ITEM="${item}" METRICS_SUBDIR="${sub}" bash "${HERE}/06_01_evaluate_ispy2_run.sh" "${rid}" "${cat}" "${tc}" &
        sleep "${SPACING_S:-5}"
    done
done
wait
echo "[$(date '+%F %T')] all duke evaluations finished"
