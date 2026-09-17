#!/usr/bin/env bash
# Paired significance testing from a YAML config (canonical shared driver).
#   bash 06_06_significance_from_config.sh configs/ispy2_t1wce_significance_01.yaml
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
CFG="${1:?usage: $0 <config.yaml>}"
if [[ "$CFG" != /* ]]; then CFG="${HERE}/${CFG}"; fi
.venv/bin/python datasets/00_commun_scripts/00_03_evaluate/significance_from_config.py "${CFG}"
