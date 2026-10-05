#!/usr/bin/env bash
# Aggregate an pansegdata results table from a YAML config via the canonical shared driver (do NOT compute ad-hoc tables or
# p-values; the inline "sig. vs ref" column is auto-wired). Configs are GENERATED from the roster by 06_05_write_configs.sh.
#   bash 06_02_aggregate_from_config.sh configs/pansegdata_t1wce_01_results.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
CFG="${1:?usage: $0 <config.yaml>}"
[[ "$CFG" != /* ]] && CFG="${HERE}/${CFG}"
.venv/bin/python benchmark/00_commun_scripts/00_03_evaluate/aggregate_from_config.py "${CFG}"
