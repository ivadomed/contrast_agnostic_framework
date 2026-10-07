#!/bin/bash
# Build SRCSM source-matched test inputs for ONE training setting (CPU job body; submit via run_job).
#
#   bash srcsm_match_setting.sh <SETTING> <TC> <TRAIN_IMAGES_DIR> <TRAIN_MODALITY> <IN_DIR:MODALITY> [...]
#
#   SETTING           label, e.g. open-ms_flair (names the CDF + the $SCRATCH work dir)
#   TC                suffix tag, e.g. flair: every input dir X gets a sibling symlink X_srcmatch_<TC>,
#                     which predict_common.sh picks up with PREDICT_INPUT_SUFFIX=_srcmatch_<TC>
#   TRAIN_IMAGES_DIR  the training dataset's imagesTr (the "source" of SRCSM's average CDF)
#   IN_DIR:MODALITY   each test input dir the model is predicted on (already FOV-cropped where the
#                     benchmark crops), with mr|ct for SRCSM's per-image preprocessing branch
#
# Matched images live on $SCRATCH/srcsm_match/<SETTING>/ (never in project space); only the symlink
# goes next to the original dir. The training contrast is in the suffix because several models share
# one input dir (e.g. BraTS T1n/T2w both read Dataset051's imagesTs_*) and each needs its own matching.
set -euo pipefail
SETTING="$1"; TC="$2"; TRAIN="$3"; TMOD="$4"; shift 4
cd /project/aip-jcohen/paulh/mri_synthesis_project
PY=".venv/bin/python -I"
TOOL=benchmark/00_commun_scripts/00_02_predict/srcsm_source_matching.py
WORK="${SCRATCH:?}/srcsm_match/${SETTING}"
mkdir -p "${WORK}"
CDF="${WORK}/average_cdf.json"
# EXCLUDE_CASES (optional env): JSON list of held-out case ids that sit inside TRAIN_IMAGES_DIR
[ -s "${CDF}" ] || ${PY} ${TOOL} cdf --images "${TRAIN}" --modality "${TMOD}" --out "${CDF}" ${EXCLUDE_CASES:+--exclude-cases "${EXCLUDE_CASES}"}
for spec in "$@"; do
  IN="${spec%:*}"; MOD="${spec##*:}"
  _rel="${IN#*/benchmark/02_tasks/}"; OUT="${WORK}/${_rel//\//__}"     # unique per input dir (every external cohort's parent is raw/)
  ${PY} ${TOOL} match --cdf "${CDF}" --in-dir "${IN}" --out-dir "${OUT}" --modality "${MOD}" --workers "${SLURM_CPUS_PER_TASK:-4}"
  ${PY} ${TOOL} qc --cdf "${CDF}" --in-dir "${IN}" --out-dir "${OUT}" --modality "${MOD}"
  LINK="${IN%/}_srcmatch_${TC}"
  if [ -L "${LINK}" ]; then ln -sfn "${OUT}" "${LINK}"; elif [ -e "${LINK}" ]; then echo "!! ${LINK} exists and is not a symlink" >&2; exit 1; else ln -s "${OUT}" "${LINK}"; fi
done
# identity sanity check: training images matched to their own average CDF must already be close to it
SELF="${WORK}/_self_check"; mkdir -p "${SELF}/in"
for f in $(ls "${TRAIN}"/*_0000.nii.gz | grep -v -F -f <(${PY} -c "import json,sys; [print(c+'_0000') for c in json.load(open(sys.argv[1]))]" "${EXCLUDE_CASES:-/dev/null}" 2>/dev/null || true) | head -3); do ln -sf "$f" "${SELF}/in/"; done
${PY} ${TOOL} match --cdf "${CDF}" --in-dir "${SELF}/in" --out-dir "${SELF}/out" --modality "${TMOD}" --overwrite
${PY} ${TOOL} qc --cdf "${CDF}" --in-dir "${SELF}/in" --out-dir "${SELF}/out" --modality "${TMOD}" || true
echo "srcsm_match_setting ${SETTING}: DONE"
