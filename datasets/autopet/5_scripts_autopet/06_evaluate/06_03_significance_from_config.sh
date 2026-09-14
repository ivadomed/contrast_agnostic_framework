#!/usr/bin/env bash
# Full paired-significance report for autopet (canonical shared driver).
#   bash 06_03_significance_from_config.sh [configs/autopet_ct_01_results.yaml]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
CFG="${1:-${HERE}/configs/autopet_ct_01_results.yaml}"
.venv/bin/python datasets/00_commun_scripts/00_03_evaluate/significance_from_config.py "${CFG}"
