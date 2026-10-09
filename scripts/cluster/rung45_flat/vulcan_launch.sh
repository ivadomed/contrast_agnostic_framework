#!/usr/bin/env bash
# VULCAN launch of ladder rung 4.5 (FLAT fill: PALETTE's partition at its region target means + PALETTE's label step, no
# noise, no texture), 2026-10-08, through the GENERATED wrappers (scripts/cluster/rung45_flat/make_wrappers.py: exact
# copy of each rung-4 lblvor wrapper with only the config/method changed). Time limits = rung 4 lblvor's (same cost).
# ROWS = the 16 paper settings; on-harmony is commented out when it trains on TamIA instead (see tamia_*.sh here).
#   bash scripts/cluster/rung45_flat/vulcan_launch.sh --dry-run | --launch | --record
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
MODE="${1:---dry-run}"
OUT="${SCRATCH:?}/rung45_flat"; mkdir -p "${OUT}"
export RUN_JOB_ACCOUNT=aip-jcohen RUN_JOB_GPU_TYPE=l40s
export RUN_JOB_EXCLUDE_NODES="${RUN_JOB_EXCLUDE_NODES:-rack02-06}"   # CUDA init failures there (2026-10-06)
T=benchmark/02_tasks
PIN_PANSEG=/project/aip-jcohen/paulh/pansegdata_auglab_7b761b5/AugLab   # pansegdata: same pinned AugLab as its rungs 2-5 (differs from
# the current AugLab only by the opt-in partial-volume code in fromSeg.py/transforms.py, pv_prob=0 -> no effect, no extra RNG)
ROWS=(
"brain_tumor/brats2024-glioma|t1n|36:00:00" "brain_tumor/brats2024-glioma|t2w|36:00:00"
"brain_tumor/brats2024-glioma|t2f|36:00:00" "brain_tumor/brats2024-glioma|t1c|36:00:00"
# on-harmony -> TamIA (rung45_flat/tamia_*): "brain_healthy/on-harmony|t1w|52:00:00" "brain_healthy/on-harmony|t2w|72:00:00" "brain_healthy/on-harmony|dwi_ap|72:00:00"
"brain_ms/open-ms|flair|30:00:00" "brain_ms/open-ms|t1w|30:00:00"
"abdomen_healthy/chaos|t1in|08:00:00" "abdomen_healthy/chaos|t2spir|08:00:00"
"mandible_healthy/toothfairy2|cbct|36:00:00"
"breast_cancer/ispy2|t1wce|30:00:00" "breast_cancer/ispy2|t2w|30:00:00"
"pelvis_healthy/totalseg-pelvic|ct|10:00:00" "pelvis_healthy/totalseg-pelvic|mri|10:00:00"
"pancreas_disease/pansegdata|t1wce|34:00:00" "pancreas_disease/pansegdata|t2w|34:00:00"   # added 2026-10-09 (rung-4 lblvor limits)
)
if [ "${MODE}" = --record ]; then
  squeue -u "$USER" -h -o '%i|%j|%T' | awk -F'|' '$2 ~ /^fold[0-9]_.*_flatfill_[0-9]+_[0-9]+$/' | sort -t'|' -k2 > "${OUT}/jobs.tsv"
  { cut -d'|' -f2 "${OUT}/jobs.tsv" | sed -E 's/^fold[0-9]_//'; cat "${OUT}/run_ids.txt" 2>/dev/null || true; } | sort -u > "${OUT}/run_ids.new" && mv "${OUT}/run_ids.new" "${OUT}/run_ids.txt"
  echo "[record] $(wc -l < "${OUT}/jobs.tsv") fold jobs, $(wc -l < "${OUT}/run_ids.txt") runs -> ${OUT}/{jobs.tsv,run_ids.txt}"; exit 0
fi
[ "${MODE}" = --dry-run ] || [ "${MODE}" = --launch ] || { echo "usage: $0 --dry-run|--launch|--record" >&2; exit 2; }
n=0; FAILED=0
for row in "${ROWS[@]}"; do IFS='|' read -r ds tag tl <<<"${row}"
  dd="$T/${ds}/5_scripts_${ds##*/}/04_train"
  w=( "${dd}"/04_*_train_"${tag}"_*_flatfill.sh )
  [ "${#w[@]}" = 1 ] && [ -f "${w[0]}" ] || { echo "ERROR: expected exactly one ${tag} *_flatfill wrapper in ${dd}" >&2; exit 1; }
  grep -q 'rung45_flat/make_wrappers.py' "${w[0]}" || { echo "ERROR: ${w[0]} is not a generated flatfill wrapper" >&2; exit 1; }
  m="$(sed -n 's/^METHOD="\(.*\)"/\1/p' "${w[0]}")"
  # ledger of launched wrappers (a job-name grep on the contrast tag matched other datasets' jobs: t2w, ct~cbct, t1wce)
  if grep -qxF "${w[0]}" "${OUT}/launched.txt" 2>/dev/null; then
    printf "[r45] %-18s %-7s already launched -- skip\n" "${ds##*/}" "${tag}"; continue; fi
  printf "[r45] %-18s %-7s %-55s time=%s\n" "${ds##*/}" "${tag}" "${w[0]##*/}" "${tl}"
  n=$((n+1)); [ "${MODE}" = --launch ] || continue
  ( export RUN_JOB_TIME_DEFAULT="${tl}"
    [ "${ds##*/}" = pansegdata ] && export CE_EXTRA_PYTHONPATH="${PIN_PANSEG}"
    bash "${w[0]}" ) < /dev/null 2>&1 | tee -a "${OUT}/launch.log" \
    && echo "${w[0]}" >> "${OUT}/launched.txt" \
    || { echo "[r45] LAUNCH FAILED: ${w[0]##*/}" | tee -a "${OUT}/launch.log"; FAILED=$((FAILED+1)); }
  sleep 3
done
echo "[r45] ${n} wrappers x 3 folds, ${FAILED} launch failures ($([ "${MODE}" = --launch ] && echo launched || echo dry run))"
[ "${MODE}" = --launch ] && bash "$0" --record || true
