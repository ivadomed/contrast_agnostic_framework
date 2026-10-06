#!/usr/bin/env bash
# Reproduce the paper's existing Open-MS NGF numbers (FLAIR) from the already-generated
# noblur volumes with the existing compute_ngf_texture.py (erode x3, foreground+lesion ROI).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
L1="${REPO}/benchmark/02_tasks/brain_ms/open-ms/7_analysis_open-ms/texture_analysis_lvl_1"
source "${REPO}/benchmark/02_tasks/brain_ms/open-ms/5_scripts_open-ms/00_utils/env.sh"
PY="${REPO}/.venv/bin/python"
export RUN_JOB_EXCLUDE_NODES="${RUN_JOB_EXCLUDE_NODES:-rack02-06}"  # node with broken CUDA (seen 2026-10-05)
run_job --name ngf_verify_oms --gpus 1 --time 01:00:00 --log "${HERE}/logs/verify_openms.log" --wait -- \
  bash -c "${PY} -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 3)' && ${PY} ${L1}/scripts/compute_ngf_texture.py --device cuda --source FLAIR --set-label flair_noblur_eroded \
   --erode-iters 3 --generated-root ${L1}/../data/generated_noblur \
   --methods palette,v26_6_2_noisefill_v2,auglab_default,synthseg_em,synthseg_noem \
   --output-csv ${HERE}/data/verify_openms_flair_eroded.csv"
