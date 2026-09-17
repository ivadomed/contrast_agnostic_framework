#!/usr/bin/env bash
# Build the toothfairy2 CBCT ladder with TWO independent CT sources (hanseg + pddca),
# CT pooled as one stratum so adding a dataset does not silently re-weight OOD toward CT.
# See the .py header. Reads the mandible-only metrics roots; run ids are recovered from
# the metrics dirs (the TamIA packs do not exist on Vulcan).
#   bash 06_11_ladder_summary_2ct.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
.venv/bin/python "${HERE}/06_11_ladder_summary_2ct.py"
