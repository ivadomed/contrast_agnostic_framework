#!/usr/bin/env bash
# Pool both isles2022 training contrasts (dwi + flair) into one table per method via the shared combined_modality_summary.py.
#   bash 06_04_combined_modality_summary.sh [configs/isles2022_combined_01_results.yaml]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
CFG="${1:-configs/isles2022_combined_01_results.yaml}"
[[ "$CFG" != /* ]] && CFG="${HERE}/${CFG}"
.venv/bin/python benchmark/00_commun_scripts/00_03_evaluate/combined_modality_summary.py "${CFG}"
