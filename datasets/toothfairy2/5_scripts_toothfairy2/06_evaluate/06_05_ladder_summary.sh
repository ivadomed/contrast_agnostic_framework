#!/usr/bin/env bash
# Build the toothfairy2 CBCT ladder (shared engine; OOD comes from the hanseg
# cross-dataset evaluator — see the .py header).
# BOTH suite packs are required: rung 1 (baseline) lives in pack A, rungs 6/7 (OURS)
# in pack B, and each pack's RUN_IDS.env lists all six ids regardless of which it
# trained.
#   bash 06_05_ladder_summary.sh <SUITE_A_PACK> <SUITE_B_PACK> <LADDER_PACK>
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
.venv/bin/python "${HERE}/06_05_ladder_summary.py" \
    --suite-pack-a "${1:?need suiteA pack}" --suite-pack-b "${2:?need suiteB pack}" \
    --ladder-pack "${3:?need ladder pack}"
