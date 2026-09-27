#!/usr/bin/env bash
# Full paired-significance report for brats2024-glioma (canonical shared driver).
# Reuses the SAME per-modality config as 06_02_aggregate_from_config.sh (they share
# the runs:/in_domain_contrast: schema) so the significance test lines up exactly
# with the headline summary table -- same pattern as chaos/toothfairy2.
#   bash 06_03_significance_from_config.sh [configs/brats_t1n_01_results.yaml] [--ref <exact run id>] [--metric dice]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
CFG="${1:-configs/brats_t1n_01_results.yaml}"
[[ "$CFG" != /* ]] && CFG="${HERE}/${CFG}"
shift || true
.venv/bin/python benchmark/00_commun_scripts/00_03_evaluate/significance_from_config.py "${CFG}" "$@"
