#!/usr/bin/env bash
# toothfairy2's `combined_contrasts` table — the per-task roll-up the cross-DATASET
# meta_task_heatmap consumes. Thin wrapper over the canonical shared driver
# datasets/00_commun_scripts/00_03_evaluate/combined_modality_summary.py.
#
# NOTE: for every other dataset this pools TWO training modalities. toothfairy2 has
# one, so the config's single `modalities:` entry pools the two TEST sources
# (own CBCT + hanseg CT) via the shared per-modality `sources:` support.
#   bash 06_07_combined_modality_summary.sh [configs/toothfairy2_combined_01_results.yaml]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
CFG="${1:-${HERE}/configs/toothfairy2_combined_01_results.yaml}"
.venv/bin/python datasets/00_commun_scripts/00_03_evaluate/combined_modality_summary.py "${CFG}"
