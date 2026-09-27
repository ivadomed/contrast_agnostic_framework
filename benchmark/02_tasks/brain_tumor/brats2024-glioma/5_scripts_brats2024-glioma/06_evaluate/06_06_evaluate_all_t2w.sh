#!/usr/bin/env bash
# Evaluate all T2w experiments listed in a config file, then aggregate.
# Run IDs are read from the config — no separate list to maintain.
#
# Usage:
#   bash 06_06_evaluate_all_t2w.sh [config.yaml]
#
# Defaults to configs/brats_t2w_01_results.yaml.
# Run this once T2w inference (05_09–05_13 predict scripts) has completed for all experiments.

set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
export TRAINING_CONTRAST="t2w"
HERE="$(cd "$(dirname "$0")" && pwd)"

CONFIG="${1:-${HERE}/configs/brats_t2w_01_results.yaml}"
[[ "$CONFIG" != /* ]] && CONFIG="${HERE}/${CONFIG}"
[ -f "$CONFIG" ] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }

mapfile -t RUNS < <(
    "${PROJECT_ROOT}/.venv/bin/python" -c \
    "import yaml,sys; [print(r) for r in yaml.safe_load(open(sys.argv[1]))['runs']]" \
    "$CONFIG"
)
[ ${#RUNS[@]} -gt 0 ] || { echo "ERROR: no runs in $CONFIG" >&2; exit 1; }

echo "[$(date '+%H:%M:%S')] Evaluating ${#RUNS[@]} T2w experiments from $(basename "$CONFIG")"

pids=()
for RUN_ID in "${RUNS[@]}"; do
    bash "${HERE}/06_01_evaluate_run.sh" "${RUN_ID}" all &
    pids+=($!)
done

any_failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || any_failed=1
done
[ "$any_failed" = "1" ] && echo "WARNING: some eval jobs failed — check *_eval.log files" >&2

echo "[$(date '+%H:%M:%S')] All T2w evaluations done — aggregating"
bash "${HERE}/06_10_aggregate_from_config.sh" "$CONFIG"
