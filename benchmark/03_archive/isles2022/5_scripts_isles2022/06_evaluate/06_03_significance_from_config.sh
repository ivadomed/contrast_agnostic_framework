#!/usr/bin/env bash
# Full paired-significance report (OOD/IND/per-contrast) for an isles2022 config via the shared significance_from_config.py.
#   bash 06_03_significance_from_config.sh configs/isles2022_dwi_significance_01.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
CFG="${1:?usage: $0 <config.yaml>}"
[[ "$CFG" != /* ]] && CFG="${HERE}/${CFG}"
.venv/bin/python benchmark/00_commun_scripts/00_03_evaluate/significance_from_config.py "${CFG}"
