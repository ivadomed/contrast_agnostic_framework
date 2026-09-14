#!/usr/bin/env bash
# autopet's `combined_contrasts` table (CT + PET pooled) — the per-task roll-up the
# cross-DATASET meta_task_heatmap consumes. Thin wrapper over the canonical shared
# driver datasets/00_commun_scripts/00_03_evaluate/combined_modality_summary.py.
#   bash 06_04_combined_modality_summary.sh [configs/autopet_combined_01_results.yaml]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
CFG="${1:-${HERE}/configs/autopet_combined_01_results.yaml}"
.venv/bin/python datasets/00_commun_scripts/00_03_evaluate/combined_modality_summary.py "${CFG}"
