#!/usr/bin/env bash
# THE HEADLINE TABLE for toothfairy2: in-domain CBCT + cross-modality hanseg CT.
# Thin wrapper over the canonical shared driver
# datasets/00_commun_scripts/00_03_evaluate/aggregate_from_config.py — the inline
# "sig. vs ref" column is auto-wired there (see CLAUDE.md); do NOT compute a second
# significance pass on top.
#   bash 06_06_cross_dataset_summary.sh [configs/toothfairy2_cross_dataset_01_results.yaml]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
CFG="${1:-${HERE}/configs/toothfairy2_cross_dataset_01_results.yaml}"
.venv/bin/python datasets/00_commun_scripts/00_03_evaluate/aggregate_from_config.py "${CFG}"
