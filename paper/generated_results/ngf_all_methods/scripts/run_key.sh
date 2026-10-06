#!/usr/bin/env bash
# One (dataset/contrast) NGF job on 1 Vulcan L40S via run_job.
# usage: run_key.sh <registry key> <time HH:MM:SS> [extra ngf_all_methods.py args...]
#   e.g. run_key.sh brats2024-glioma/t1n 02:00:00 --n-scans 20 --n-draws 5 --variants noblur,asblur
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
KEY="$1"; TIME="$2"; shift 2
source "${REPO}/benchmark/02_tasks/brain_ms/open-ms/5_scripts_open-ms/00_utils/env.sh"   # run_job + venv env
PY="${REPO}/.venv/bin/python"
export RUN_JOB_EXCLUDE_NODES="${RUN_JOB_EXCLUDE_NODES:-rack02-06}"   # broken CUDA node seen 2026-10-05
TAG="${KEY//\//__}"
OUTDIR="${NGF_OUTDIR:-${HERE}/data}"
mkdir -p "${OUTDIR}" "${HERE}/logs"
run_job --name "ngf_${TAG}" --gpus 1 --time "${TIME}" --log "${HERE}/logs/${TAG}${NGF_TAG_SUFFIX:-}.log" ${NGF_WAIT:+--wait} -- \
  bash -c "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; ${PY} ${HERE}/scripts/ngf_all_methods.py --key ${KEY} --out ${OUTDIR}/ngf_${TAG}${NGF_TAG_SUFFIX:-}.csv $*"
