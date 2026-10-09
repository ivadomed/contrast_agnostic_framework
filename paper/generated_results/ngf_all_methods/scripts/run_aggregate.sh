#!/usr/bin/env bash
# Aggregate data/ngf_*.csv -> ngf_all_methods.csv, ngf_summary.md, ngf_vs_dice_gain*.png (small CPU job).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
source "${REPO}/benchmark/02_tasks/brain_ms/open-ms/5_scripts_open-ms/00_utils/env.sh"
run_job --name ngf_aggregate --gpus 0 --cpus 2 --mem 8G --time 00:20:00 --wait --log "${HERE}/logs/aggregate.log" -- \
  "${REPO}/.venv/bin/python" "${HERE}/scripts/aggregate_ngf.py"
cat "${HERE}/logs/aggregate.log" | tail -5
