#!/usr/bin/env bash
# THE HEADLINE TABLE for toothfairy2: in-domain CBCT + cross-modality hanseg CT.
# Thin wrapper over the canonical shared driver
# benchmark/00_commun_scripts/00_03_evaluate/aggregate_from_config.py — the inline
# "sig. vs ref" column is auto-wired there (see the project notes); do NOT compute a second
# significance pass on top.
#   bash 06_06_cross_dataset_summary.sh [configs/toothfairy2_cross_dataset_01_results.yaml]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
CFG="${1:-configs/toothfairy2_cross_dataset_01_results.yaml}"
[[ "$CFG" != /* ]] && CFG="${HERE}/${CFG}"
.venv/bin/python benchmark/00_commun_scripts/00_03_evaluate/aggregate_from_config.py "${CFG}"
