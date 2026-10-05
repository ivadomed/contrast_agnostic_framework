#!/usr/bin/env bash
# Training status on Killarney: for every run x fold of the REAL results base, the last epoch reached (fold_N/training_log_*.txt: real progress lives there, not in
# the sbatch stdout), whether checkpoint_final.pth exists, plus the state of this user's pansegdata fold jobs. Flags folds whose log never reached epoch 1.
#   bash 04_29_killarney_status.sh [--problems]     # --problems: only folds with no progress / not running and not final
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
ONLY=0; [ "${1:-}" = "--problems" ] && ONLY=1
BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}"
echo "jobs (this user, pansegdata folds):"; squeue -u "$USER" -h -o "%T" --name="$(squeue -u "$USER" -h -o %j | grep -E '^fold[0-9]_pansegdata' | sort -u | paste -sd, -)" 2>/dev/null | sort | uniq -c | sed 's/^/   /' || true
printf "%-6s %-8s %-62s %5s %7s %s\n" contrast fold run last_epoch final note
for FD in $(find "${BASE}" -mindepth 6 -maxdepth 6 -type d -name 'fold_*' 2>/dev/null | sort); do
    run="$(echo "${FD}" | sed -E "s#${BASE}/##; s#/Dataset[^/]*/[^/]*/fold_.*##")"; fold="${FD##*/fold_}"; con="${run%%/*}"
    L="$(ls -t "${FD}"/training_log_*.txt 2>/dev/null | head -1 || true)"
    ep="-"; [ -n "${L}" ] && ep="$(grep -oE 'Epoch [0-9]+ *$' "${L}" 2>/dev/null | tail -1 | awk '{print $2}' || true)"; ep="${ep:--}"
    fin="no"; [ -f "${FD}/checkpoint_final.pth" ] && fin="yes"
    note=""; { [ "${ep}" = "-" ] || [ "${ep}" = "0" ]; } && [ "${fin}" = "no" ] && note="NO PROGRESS YET"
    [ "${ONLY}" = 1 ] && [ -z "${note}" ] && continue
    printf "%-6s %-8s %-62s %5s %7s %s\n" "${con}" "${fold}" "${run#*/}" "${ep}" "${fin}" "${note}"
done
