#!/usr/bin/env bash
# Summarise a sizing probe (04_24): per fold the median of the later epochs' "Epoch time" from nnU-Net's training_log (the real progress log,
# not the pack stdout), the extrapolated NNUNET_NUM_EPOCHS-epoch wall time, and what failed. GPU memory: see the pack job's nvidia-smi samples
# (probe_gpu_mem.log in the pack dir if the sampler ran).   bash 04_25_probe_report.sh <PACK_DIR> [EPOCHS_TARGET=2000]
set -uo pipefail
PACK="${1:?pack dir}"; TARGET="${2:-2000}"
PROBE="$(dirname "${PACK}")"
printf "%-52s %8s %8s %12s %s\n" "run (fold0)" "epochs" "med s/ep" "h per ${TARGET}ep" "status"
for log in $(ls "${PROBE}"/01_predictions/pansegdata_model/*/*/*/*/*/fold_0/training_log_*.txt 2>/dev/null); do
    run="$(echo "$log" | sed -E 's#.*/pansegdata_model/([^/]+)/[^/]+/([^/]+)/.*#\1/\2#')"
    awk -v run="$run" -v T="$TARGET" '/Epoch time:/ {t[++n]=$(NF-1)} END {
        if (n<4) {printf "%-52s %8d %8s %12s %s\n", run, n, "-", "-", "too few epochs"; exit}
        s=int(n/3)+1; m=0; for(i=s;i<=n;i++) a[++m]=t[i]; asort(a); med=a[int((m+1)/2)];
        printf "%-52s %8d %8.1f %12.1f %s\n", run, n, med, med*T/3600, "ok"}' "$log"
done
echo; echo "pack stdout (failures):"; grep -h "FAIL\|OK  " "${PACK}"/pack_*.out 2>/dev/null | head -20
[ -f "${PACK}/probe_gpu_mem.log" ] && { echo; echo "peak per-process GPU memory (MiB):"; sort -t, -k3 -n "${PACK}/probe_gpu_mem.log" | tail -3; }
