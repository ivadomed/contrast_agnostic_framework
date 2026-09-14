#!/usr/bin/env bash
# Aggregate an autopet results table from a YAML config (canonical shared driver — do
# NOT compute ad-hoc tables or p-values; the inline "sig. vs ref" column is auto-wired).
#   bash 06_02_aggregate_from_config.sh [configs/autopet_ct_01_results.yaml]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
CFG="${1:-${HERE}/configs/autopet_ct_01_results.yaml}"
.venv/bin/python datasets/00_commun_scripts/00_03_evaluate/aggregate_from_config.py "${CFG}"
