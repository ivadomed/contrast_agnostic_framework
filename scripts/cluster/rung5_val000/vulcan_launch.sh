#!/usr/bin/env bash
# VULCAN launch of the val000 ladder real-fill rungs (2026-10-05/06): rung 5 "PALETTE alone" for 18 settings and the
# rung-6 "PALETTE + boundary PV alone" for the 14 settings that have a PV branch -- 3 folds each, one L40S per fold,
# through each setting's own GENERATED wrapper (scripts/cluster/rung5_val000/make_wrappers.py: an exact copy of the
# noise-fill rung-4 wrapper with only the fill changed). Why: the previous real-fill wrappers used ValSynth trainers
# that choose checkpoint_best on SYNTHETIC validation images; every other rung chooses it on real ones.
#
#   bash scripts/cluster/rung5_val000/vulcan_launch.sh --dry-run [real|pv|all]
#   bash scripts/cluster/rung5_val000/vulcan_launch.sh --launch  [real|pv|all]
#   bash scripts/cluster/rung5_val000/vulcan_launch.sh --record                 # RUN_IDs + job ids of what is queued/running -> runs.tsv
# Time limits: old real-fill runs' s/epoch x epochs x ~1.6 margin (unknown cost -> generous), PV +15%; all under the
# 3-day tier. A fold that hits its limit resumes by re-running its wrapper with the SAME RUN_ID (train_common.sh).
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
MODE="${1:---dry-run}"; SET="${2:-all}"
OUT="${SCRATCH:?}/rung5_val000"; mkdir -p "${OUT}"
export RUN_JOB_ACCOUNT=aip-jcohen RUN_JOB_GPU_TYPE=l40s
T=benchmark/02_tasks
PIN_PANSEG=/project/aip-jcohen/paulh/pansegdata_auglab_7b761b5/AugLab   # pansegdata: same pinned AugLab code as its other Vulcan rungs

# dataset dir | contrast tag in the wrapper name | real-fill limit | PV limit ("-" = no PV branch)
ROWS=(
"brain_tumor/brats2024-glioma|t1n|36:00:00|40:00:00"
"brain_tumor/brats2024-glioma|t2w|36:00:00|40:00:00"
"brain_tumor/brats2024-glioma|t2f|36:00:00|40:00:00"
"brain_tumor/brats2024-glioma|t1c|36:00:00|40:00:00"
"brain_healthy/on-harmony|t1w|52:00:00|60:00:00"
"brain_healthy/on-harmony|t2w|52:00:00|60:00:00"
"brain_healthy/on-harmony|dwi_ap|52:00:00|60:00:00"
"brain_ms/open-ms|t1w|30:00:00|34:00:00"
"abdomen_healthy/chaos|t2spir|08:00:00|10:00:00"
"mandible_healthy/toothfairy2|cbct|36:00:00|40:00:00"
"breast_cancer/ispy2|t1wce|30:00:00|34:00:00"
"breast_cancer/ispy2|t2w|30:00:00|34:00:00"
"pelvis_healthy/totalseg-pelvic|ct|10:00:00|12:00:00"
"pelvis_healthy/totalseg-pelvic|mri|10:00:00|12:00:00"
"brain_stroke/isles2022|dwi|40:00:00|-"
"brain_stroke/isles2022|flair|40:00:00|-"
"pancreas_disease/pansegdata|t1wce|34:00:00|-"
"pancreas_disease/pansegdata|t2w|34:00:00|-"
)

if [ "${MODE}" = --record ]; then
  # job name = fold<k>_<RUN_ID> (train_common.sh); RUN_ID ends in _<YYYYMMDD>_<HHMMSS>
  squeue -u "$USER" -h -o '%i|%j|%T' | awk -F'|' '$2 ~ /^fold[0-9]_.*_v26_6_2(_pv)?_train050_val000_[0-9]+_[0-9]+$/ && $2 !~ /auglabAug/' \
    | sort -t'|' -k2 > "${OUT}/jobs.tsv"
  cut -d'|' -f2 "${OUT}/jobs.tsv" | sed -E 's/^fold[0-9]_//' | sort -u > "${OUT}/run_ids.txt"
  echo "[record] $(wc -l < "${OUT}/jobs.tsv") fold jobs, $(wc -l < "${OUT}/run_ids.txt") runs -> ${OUT}/{jobs.tsv,run_ids.txt}"
  exit 0
fi
[ "${MODE}" = --dry-run ] || [ "${MODE}" = --launch ] || { echo "usage: $0 --dry-run|--launch [real|pv|all] | --record" >&2; exit 2; }

n=0; FAILED=0
for row in "${ROWS[@]}"; do IFS='|' read -r ds tag tl_real tl_pv <<<"${row}"
  dd="$T/${ds}/5_scripts_${ds##*/}/04_train"
  for v in real pv; do
    [ "${SET}" = all ] || [ "${SET}" = "$v" ] || continue
    if [ "$v" = real ]; then m=v26_6_2_train050_val000; tl="${tl_real}"; else m=v26_6_2_pv_train050_val000; tl="${tl_pv}"; fi
    [ "${tl}" = - ] && continue
    w=( "${dd}"/04_*_train_"${tag}"_"${m}".sh )
    [ "${#w[@]}" = 1 ] && [ -f "${w[0]}" ] || { echo "ERROR: expected exactly one ${tag} ${m} wrapper in ${dd}" >&2; exit 1; }
    grep -q '^METHOD="'"${m}"'"' "${w[0]}" || { echo "ERROR: ${w[0]} METHOD is not ${m}" >&2; exit 1; }
    # already queued/running (re-runs after a partial launch only add what is missing)
    if squeue -u "$USER" -h -o '%j' | grep -qiE "^fold[0-9]_${ds##*/}_${tag}_${m}_[0-9]+_[0-9]+$"; then
      printf "[r5] %-32s %-7s %-28s already queued -- skip\n" "${ds##*/}" "${tag}" "${m}"; continue; fi
    printf "[r5] %-32s %-7s %-28s time=%s  %s\n" "${ds##*/}" "${tag}" "${m}" "${tl}" "${w[0]##*/}"
    n=$((n+1))
    [ "${MODE}" = --launch ] || continue
    ( export RUN_JOB_TIME_DEFAULT="${tl}"
      [ "${ds##*/}" = pansegdata ] && export CE_EXTRA_PYTHONPATH="${PIN_PANSEG}"
      bash "${w[0]}" ) < /dev/null 2>&1 | tee -a "${OUT}/launch.log" \
      || { echo "[r5] LAUNCH FAILED: ${w[0]##*/} (see ${OUT}/launch.log); continuing" | tee -a "${OUT}/launch.log"; FAILED=$((FAILED+1)); }
    sleep 3
  done
done
echo "[r5] ${n} wrappers x 3 folds, ${FAILED} launch failures ($([ "${MODE}" = --launch ] && echo launched || echo dry run))"
[ "${MODE}" = --launch ] && bash "$0" --record || true
