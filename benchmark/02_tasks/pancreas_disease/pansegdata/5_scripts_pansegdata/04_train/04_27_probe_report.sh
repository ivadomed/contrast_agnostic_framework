#!/usr/bin/env bash
# Report of a Killarney sizing probe: per run, median epoch time (epochs >= 3, the first epochs include warm-up) and the projected wall time of EPOCHS epochs.
# Real progress is in fold_N/training_log_*.txt (not in the sbatch stdout).   bash 04_27_probe_report.sh <probe dir> [EPOCHS=2000]
set -euo pipefail
PROBE="${1:?usage: 04_27_probe_report.sh <probe dir> [EPOCHS]}"; EPOCHS="${2:-2000}"
printf "%-62s %7s %9s %9s\n" "run (contrast/category/run id)" "epochs" "med s/ep" "${EPOCHS}-ep h"
for L in $(find "${PROBE}/01_predictions" -name 'training_log_*.txt' 2>/dev/null | sort); do
    run="$(echo "$L" | sed -E "s#.*/pansegdata_model/##; s#/Dataset[^/]*/.*##")"
    awk -v run="$run" -v E="$EPOCHS" '/Epoch time:/ {n++; t=$0; sub(/.*Epoch time: /,"",t); sub(/ s.*/,"",t); if (n>=3) v[++k]=t+0}
        END {if (k==0) {printf "%-62s %7d %9s %9s\n", run, n, "n/a", "n/a"; exit}
             asort(v); m=v[int((k+1)/2)]; printf "%-62s %7d %9.1f %9.1f\n", run, n, m, m*E/3600}' "$L"
done
echo; echo "sacct (CPU efficiency / peak RSS per probe job; your jobs only):"
sacct -u "$USER" -S today --name=$(basename "${PROBE%/}" | sed 's/^_probe_killarney_//') -X -o JobName%40,State,Elapsed,AllocCPUS,MaxRSS 2>/dev/null | head -20 || true
