#!/usr/bin/env bash
# Aggregate a healthy-spine-tum results table from a YAML config (canonical shared
# driver — do NOT compute ad-hoc tables or p-values; the inline "sig. vs ref" column
# is auto-wired).
#   bash 06_01_aggregate_from_config.sh configs/healthy-spine-tum_ct_01_results.yaml
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
CFG="${1:?usage: 06_01_aggregate_from_config.sh <configs/*.yaml>}"
[[ "$CFG" != /* ]] && CFG="${HERE}/${CFG}"
.venv/bin/python benchmark/00_commun_scripts/00_03_evaluate/aggregate_from_config.py "${CFG}"
