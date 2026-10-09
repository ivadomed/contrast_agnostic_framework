#!/usr/bin/env bash
# Predict the whole pansegdata roster (source contrasts: t1wce t2w; every method pinned in the SOURCE's roster_run_ids.tsv, incl. ladder rungs) on totalsegmri-pancreas's items. No timestamps here.
#   bash 05_02_run_all_predict.sh        env: ROSTER_ONLY="m1 m2", ROSTER_SKIP_MISSING=1, CHECKPOINT=checkpoint_final.pth
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
export SOURCE_PREFIX="PANSEG" SOURCE_CONTRASTS="t1wce t2w"
export PREDICT_SHIM="$(cd "$(dirname "$0")" && pwd)/05_01_predict_pansegdata_common.sh"
source "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_02_predict/run_all_predict_cross_common.sh"
