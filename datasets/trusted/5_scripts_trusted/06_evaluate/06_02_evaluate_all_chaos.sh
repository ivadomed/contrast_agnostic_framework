#!/usr/bin/env bash
# Evaluate ALL chaos-trained models on TRUSTED CT+US — both contrast branches.
# Thin entry point: runs the T1in batch then the T2spir batch (each fans its 6
# methods × 4 folds in parallel). See 06_01_evaluate_run.sh for single-run usage.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "=================================================================="
echo " TRUSTED <- CHAOS models | evaluate t1in + t2spir | kidney"
echo "=================================================================="

bash "${HERE}/06_04_evaluate_t1in_all.sh"
bash "${HERE}/06_05_evaluate_t2spir_all.sh"

echo ""
echo "[$(date '+%H:%M:%S')] All (t1in + t2spir) evaluations done."
