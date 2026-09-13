#!/usr/bin/env bash
# "Does the method that LEADS us actually beat us significantly?"
#
# The `sig. vs ref` column every generated table carries is ONE-SIDED ("ref is
# better"), so a p of 1.0 there means only "we cannot show OURS is better" --
# it says nothing about whether the competitor's lead is real. This driver runs
# paper/scripts/compute_reverse_significance.py, which tests the other
# direction, on one or more aggregation configs.
#
# Exists because that script refuses (loudly, by design) to run on a config
# containing an unexpanded ${METRICS_ROOT}: those vars are set by each dataset's
# own 00_utils/env.sh, and silently proceeding on an unset one is what once
# created a directory literally named '${METRICS_ROOT}' full of misfiled
# results. This wrapper sources the right env.sh per dataset so callers never
# have to remember to.
#
#   bash scripts/evaluate/run_reverse_significance.sh <dataset> <config.yaml> [<config.yaml> ...]
#   bash scripts/evaluate/run_reverse_significance.sh ispy2 \
#        datasets/ispy2/5_scripts_ispy2/06_evaluate/configs/ispy2_combined_01_results.yaml
#
# Read-only: loads existing per-case CSVs and prints. Writes nothing.
set -euo pipefail
: "${PROJECT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PROJECT_ROOT

if [ $# -lt 2 ]; then
    sed -n '2,20p' "${BASH_SOURCE[0]}" >&2
    exit 1
fi
DATASET="$1"; shift

ENV_SH="${PROJECT_ROOT}/datasets/${DATASET}/5_scripts_${DATASET}/00_utils/env.sh"
[ -f "${ENV_SH}" ] || { echo "No env.sh for dataset '${DATASET}': ${ENV_SH}" >&2; exit 1; }
set -a; source "${ENV_SH}"; set +a

cd "${PROJECT_ROOT}"
.venv/bin/python paper/scripts/compute_reverse_significance.py "$@"
