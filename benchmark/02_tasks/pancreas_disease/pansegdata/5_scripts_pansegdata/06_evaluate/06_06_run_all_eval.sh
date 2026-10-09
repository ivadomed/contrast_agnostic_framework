#!/usr/bin/env bash
# Evaluate the whole pansegdata roster for one or both training contrasts: headline methods + OURS val100 mirror -> <contrast>/,
# ladder rungs 2-5 -> <contrast>/ablations/ (automatic). RUN_IDs come from the roster (shared run_all_evaluate_common.sh).
#   bash 06_06_run_all_eval.sh [t1wce|t2w|all]      env: ROSTER_SKIP_MISSING=1, ROSTER_ONLY="m1 m2", CKPT_TAG=final
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHICH="${1:-all}"
for C in t1wce t2w; do
    [ "${WHICH}" = "all" ] || [ "${WHICH}" = "${C}" ] || continue
    (
        ENVF="env.sh"; [ "${C}" = "t2w" ] && ENVF="env_t2w.sh"
        source "${HERE}/../00_utils/${ENVF}"
        PRED="${HERE}/../05_predict"
        METHOD_SCRIPTS=( "${PRED}"/05_*_predict_${C}_*.sh )
        EVAL_SCRIPT="${HERE}/06_01_evaluate_run.sh"
        source "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_03_evaluate/run_all_evaluate_common.sh"
    )
done
