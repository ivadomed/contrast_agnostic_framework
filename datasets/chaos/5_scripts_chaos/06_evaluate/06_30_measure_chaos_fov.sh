#!/usr/bin/env bash
# Measure the CHAOS FOV relative to the kidneys/liver over the CHAOS training GT.
# Writes chaos_fov_margins.json (machine contract read by the amos/sliver07/trusted
# FOV-restricted evaluators) + a human-readable distribution report under
# 8_results_chaos/03_aggregated_results/chaos_fov_margins.md.
#
# Tiny job (32 small MR label volumes) — CPU only, routed through run_job.
#   bash 06_30_measure_chaos_fov.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_job --name "chaos_measure_fov" --gpus 0 --slot 0 --mem 8G --time 00:20:00 --wait -- \
    .venv/bin/python "${HERE}/06_30_measure_chaos_fov.py"

echo "[$(date '+%H:%M:%S')] done → ${HERE}/chaos_fov_margins.json"
