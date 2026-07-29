#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ $# -ne 1 ]; then echo "Usage: $0 <config.yaml>" >&2; exit 1; fi
CONFIG="$1"
if [[ "$CONFIG" != /* ]]; then CONFIG="${HERE}/${CONFIG}"; fi
echo "[$(date '+%H:%M:%S')] aggregating from ${CONFIG}"
.venv/bin/python "${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/aggregate_from_config.py" "${CONFIG}"
echo "[$(date '+%H:%M:%S')] done"
