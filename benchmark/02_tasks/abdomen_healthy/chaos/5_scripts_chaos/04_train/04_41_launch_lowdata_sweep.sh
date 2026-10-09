#!/usr/bin/env bash
# Low-data-regime benchmark — launch the FULL sweep: 6 methods × 5 regimes × 4 folds
# = 120 single-GPU jobs, each a 3 h allocation. All launched in parallel (Vulcan
# schedules them as GPUs free up). Fire-and-exit.
#
# Usage:
#   bash 04_41_launch_lowdata_sweep.sh                 # full grid
#   bash 04_41_launch_lowdata_sweep.sh baseline v26_6_2   # subset of methods
#   REGIMES="2 12" bash 04_41_launch_lowdata_sweep.sh baseline   # subset of regimes
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

METHODS=("$@")
if [ "${#METHODS[@]}" -eq 0 ]; then
    METHODS=(baseline v26_6_2 auglab_default synthseg_EM synthseg_noEM auglabAug_v26_6_2)
fi
read -ra REGIME_LIST <<< "${REGIMES:-1 2 4 8 12}"

echo "[$(date '+%H:%M:%S')] === Low-data sweep: ${#METHODS[@]} methods × ${#REGIME_LIST[@]} regimes (4 folds each) ==="
echo "  methods: ${METHODS[*]}"
echo "  regimes: ${REGIME_LIST[*]}"
echo ""

for method in "${METHODS[@]}"; do
    for N in "${REGIME_LIST[@]}"; do
        echo "[$(date '+%H:%M:%S')] launching ${method} N=${N}"
        bash "${HERE}/04_40_train_lowdata.sh" "${method}" "${N}" &
        sleep 3   # gentle spacing so 120 sbatch calls don't arrive all at once
    done
done
wait

echo ""
echo "[$(date '+%H:%M:%S')] === All low-data training jobs submitted ==="
echo "  monitor: squeue -u \$USER   |   logs: /tmp/nnunet_chaos_lowdata_*/fold*.log"
