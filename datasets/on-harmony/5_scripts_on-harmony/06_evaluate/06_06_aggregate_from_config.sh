#!/usr/bin/env bash
# Aggregate an on-harmony benchmark config into cross-contrast comparison tables + heatmaps,
# using the SHARED config-driven aggregator (same as chaos/brats — no on-harmony-specific
# aggregation code). Reads <metrics_dir>/<category>_<run_id>/fold*/eval_all.csv for every run
# in the config (standard 02_metrics layout) and writes <metrics_dir>/<output_prefix>_summary.md
# + _heatmap_{dice,hd95}.png. Run 06_01_evaluate_testset.sh first to produce the eval_all.csv.
#
# Usage:
#   bash 06_06_aggregate_from_config.sh configs/on-harmony_T1w.yaml
#   bash 06_06_aggregate_from_config.sh configs/on-harmony_T2w.yaml
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
HERE="$(cd "$(dirname "$0")" && pwd)"

CFG="${1:?Usage: $0 <config.yaml>}"
[[ "$CFG" != /* ]] && CFG="${HERE}/${CFG}"
[ -f "$CFG" ] || { echo "ERROR: config not found: $CFG" >&2; exit 1; }

.venv/bin/python "${PROJECT_ROOT}/scripts/evaluate/aggregate_from_config.py" "${CFG}"
