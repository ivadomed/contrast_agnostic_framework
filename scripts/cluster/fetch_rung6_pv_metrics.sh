#!/usr/bin/env bash
# Pull the rung-6 (PALETTE + boundary PV) METRICS run dirs from TamIA scratch into this (Vulcan) repo, to
# the exact paths the ladders read their rung-5 metrics from. Targeted (only the new run dirs) -- NOT
# fetch_tamia_results.sh, which re-pulls a whole 02_metrics tree over Vulcan-side files.
# The mandible arms are PROMOTED the same way their rung 5 was (verified byte-identical 2026-10-04):
# TamIA mandible_only[_sif]/ablations/<run> -> Vulcan cbct/ablations/<run>.
#   bash scripts/cluster/fetch_rung6_pv_metrics.sh [open-ms toothfairy2 hanseg pddca totalseg-pelvic]
# Verifies every file by md5 (no short transfers).
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
T=benchmark/02_tasks; SC=/scratch/p/paulh
SEL="${*:-open-ms toothfairy2 hanseg pddca totalseg-pelvic}"
TS=20261003_111216; TS_OMS=20261003_111215; TS_PEL=20261003_120315
# <dataset>|TamIA metrics dir (under ${SC})|Vulcan metrics dir (under ${T})
ROWS=(
"open-ms|open-ms/8_results_open-ms/02_metrics/open_ms_model/flair/ablations/nnUNet_open-ms_flair_v26_6_2_pv_train050_val000_${TS_OMS}|brain_ms/open-ms/8_results_open-ms/02_metrics/open_ms_model/flair/ablations"
"open-ms|open-ms/8_results_open-ms/02_metrics/open_ms_model/t1w/ablations/nnUNet_open-ms_t1w_v26_6_2_pv_train050_val100_${TS_OMS}|brain_ms/open-ms/8_results_open-ms/02_metrics/open_ms_model/t1w/ablations"
"toothfairy2|toothfairy2/8_results/02_metrics/toothfairy2_model/cbct_mandible_only/ablations/nnUNet_toothfairy2_cbct_v26_6_2_pv_train050_val100_${TS}|mandible_healthy/toothfairy2/8_results_toothfairy2/02_metrics/toothfairy2_model/cbct/ablations"
"hanseg|hanseg/8_results/02_metrics/toothfairy2_model/cbct/mandible_only_sif/ablations/nnUNet_toothfairy2_cbct_v26_6_2_pv_train050_val100_${TS}|mandible_healthy/hanseg/8_results_hanseg/02_metrics/toothfairy2_model/cbct/ablations"
"pddca|pddca/8_results/02_metrics/toothfairy2_model/cbct/mandible_only_sif/ablations/nnUNet_toothfairy2_cbct_v26_6_2_pv_train050_val100_${TS}|mandible_healthy/pddca/8_results_pddca/02_metrics/toothfairy2_model/cbct/ablations"
"totalseg-pelvic|totalseg-pelvic/8_results_totalseg-pelvic/02_metrics/totalseg_pelvic_model/ct/ablations/nnUNet_totalseg-pelvic_ct_v26_6_2_pv_train050_val100_${TS_PEL}|pelvis_healthy/totalseg-pelvic/8_results_totalseg-pelvic/02_metrics/totalseg_pelvic_model/ct/ablations"
"totalseg-pelvic|totalseg-pelvic/8_results_totalseg-pelvic/02_metrics/totalseg_pelvic_model/mri/ablations/nnUNet_totalseg-pelvic_mri_v26_6_2_pv_train050_val100_${TS_PEL}|pelvis_healthy/totalseg-pelvic/8_results_totalseg-pelvic/02_metrics/totalseg_pelvic_model/mri/ablations"
)
for r in "${ROWS[@]}"; do IFS='|' read -r ds src dst <<<"$r"
  [[ " ${SEL} " == *" ${ds} "* ]] || continue
  run="$(basename "${src}")"; sdir="$(dirname "${src}")"; mkdir -p "${T}/${dst}"
  ssh tamia.alliancecan.ca "cd '${SC}/${sdir}' && tar cf - '${run}'" | tar xf - -C "${T}/${dst}"
  a=$(cd "${T}/${dst}/${run}" && find . -type f | LC_ALL=C sort | xargs md5sum | md5sum | cut -c1-32)
  b=$(ssh tamia.alliancecan.ca "cd '${SC}/${src}' && find . -type f | LC_ALL=C sort | xargs md5sum | md5sum | cut -c1-32")
  n=$(find "${T}/${dst}/${run}" -type f | wc -l)
  [ "$a" = "$b" ] && echo "[fetch] OK ${n} files  ${ds}: ${run}" || { echo "[fetch] MISMATCH ${ds}: ${run}"; exit 1; }
done
