#!/usr/bin/env bash
# Evaluate every run in an on-harmony benchmark config (cross-contrast test set),
# by running 06_01_evaluate_testset.sh for each run_id. Each 06_01 invocation fans out one
# GPU job per fold and writes predictions + metrics into the STANDARD co-located layout
# (01_predictions/.../<RUN_ID>/fold{k}/ and 02_metrics/.../<cat>_<RUN_ID>/fold{k}/eval_all.csv).
# Run this once the runs have finished training; then aggregate with 06_06.
# Set ONHEVAL_TIME (e.g. 00:45:00) to cap each per-fold GPU job.
#
# Usage:
#   bash 06_05_evaluate_from_config.sh configs/on-harmony_T1w.yaml
#   bash 06_05_evaluate_from_config.sh configs/on-harmony_T2w.yaml
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
HERE="$(cd "$(dirname "$0")" && pwd)"

CFG="${1:?Usage: $0 <config.yaml>}"
[[ "$CFG" != /* ]] && CFG="${HERE}/${CFG}"
[ -f "$CFG" ] || { echo "ERROR: config not found: $CFG" >&2; exit 1; }

mapfile -t RUNS < <(.venv/bin/python -c "import yaml; print('\n'.join(yaml.safe_load(open('${CFG}'))['runs']))")
echo "[$(date '+%H:%M:%S')] evaluating ${#RUNS[@]} runs from $(basename "$CFG")"
for r in "${RUNS[@]}"; do
    echo "──────── $r ────────"
    bash "${HERE}/06_01_evaluate_testset.sh" "$r" || echo "  ! eval failed/incomplete for $r (continuing)"
done
echo "[$(date '+%H:%M:%S')] done. Aggregate with: bash 06_06_aggregate_from_config.sh $(basename "$CFG")"
