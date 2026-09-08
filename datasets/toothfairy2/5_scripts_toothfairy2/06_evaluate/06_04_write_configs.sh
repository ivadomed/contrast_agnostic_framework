#!/usr/bin/env bash
# Generate the four results configs from the packs' recorded RUN_IDs.
# The 6-method suite is split across two packs and each pack's RUN_IDS.env lists all
# six ids regardless of which three it trained — so BOTH must be passed (see the .py).
#   bash 06_04_write_configs.sh <SUITE_A_PACK> <SUITE_B_PACK> [LADDER_PACK]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
A="${1:?need suiteA pack dir}"; B="${2:?need suiteB pack dir}"; LADDER="${3:-}"
.venv/bin/python "${HERE}/06_04_write_configs.py" \
    --suite-pack-a "${A}" --suite-pack-b "${B}" ${LADDER:+--ladder-pack "${LADDER}"}
