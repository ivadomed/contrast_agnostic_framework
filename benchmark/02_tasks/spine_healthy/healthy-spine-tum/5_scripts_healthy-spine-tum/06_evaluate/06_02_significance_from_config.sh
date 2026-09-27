#!/usr/bin/env bash
# Full paired-significance report for healthy-spine-tum (canonical shared driver).
#   bash 06_02_significance_from_config.sh configs/healthy-spine-tum_ct_significance_01.yaml
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
CFG="${1:?usage: 06_02_significance_from_config.sh <configs/*.yaml>}"
[[ "$CFG" != /* ]] && CFG="${HERE}/${CFG}"
.venv/bin/python benchmark/00_commun_scripts/00_03_evaluate/significance_from_config.py "${CFG}"
