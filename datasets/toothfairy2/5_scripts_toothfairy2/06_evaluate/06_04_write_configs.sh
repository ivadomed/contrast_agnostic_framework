#!/usr/bin/env bash
# Generate the results configs from the packs' recorded RUN_IDs.
#   bash 06_04_write_configs.sh <SUITE_PACK_DIR> [LADDER_PACK_DIR]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
SUITE="${1:?need suite pack dir}"; LADDER="${2:-}"
.venv/bin/python "${HERE}/06_04_write_configs.py" --suite-pack "${SUITE}" \
    ${LADDER:+--ladder-pack "${LADDER}"}
