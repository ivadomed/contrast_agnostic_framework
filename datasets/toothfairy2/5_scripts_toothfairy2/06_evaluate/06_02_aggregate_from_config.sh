#!/usr/bin/env bash
# Aggregate the toothfairy2 CBCT results table from a YAML config (canonical shared
# driver — do NOT compute ad-hoc tables or p-values; the inline "sig. vs ref" column
# is auto-wired, see CLAUDE.md).
#   bash 06_02_aggregate_from_config.sh [configs/toothfairy2_cbct_01_results.yaml]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
CFG="${1:-${HERE}/configs/toothfairy2_cbct_01_results.yaml}"
.venv/bin/python datasets/00_commun_scripts/00_03_evaluate/aggregate_from_config.py "${CFG}"
