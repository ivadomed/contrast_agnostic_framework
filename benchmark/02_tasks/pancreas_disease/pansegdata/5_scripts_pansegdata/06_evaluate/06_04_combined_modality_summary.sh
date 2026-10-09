#!/usr/bin/env bash
# Pool both pansegdata training contrasts (t1wce + t2w) into one table per method via the shared combined_modality_summary.py.
#   bash 06_04_combined_modality_summary.sh [configs/pansegdata_combined_01_results.yaml]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
CFG="${1:-configs/pansegdata_combined_01_results.yaml}"
[[ "$CFG" != /* ]] && CFG="${HERE}/${CFG}"
.venv/bin/python benchmark/00_commun_scripts/00_03_evaluate/combined_modality_summary.py "${CFG}"
