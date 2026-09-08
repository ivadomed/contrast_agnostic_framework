#!/usr/bin/env bash
# Build the toothfairy2 CBCT ladder (shared engine; OOD comes from the hanseg
# cross-dataset evaluators — see the .py header).
#   bash 06_05_ladder_summary.sh <SUITE_PACK_DIR> <LADDER_PACK_DIR>
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
.venv/bin/python "${HERE}/06_05_ladder_summary.py" \
    --suite-pack "${1:?need suite pack dir}" --ladder-pack "${2:?need ladder pack dir}"
