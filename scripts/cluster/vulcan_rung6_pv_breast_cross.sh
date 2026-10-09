#!/usr/bin/env bash
# Ladder RUNG 6 (PALETTE + boundary PV) ispy2 models -> the breast CROSS-DATASET test sets
# (duke-breast-mri *_uniap, ispy1, acrin6698 dwi_uniap). Runs on VULCAN because those test sets
# (A-P crops, ispy1, acrin6698) exist only here -- same as their rung 5 (2026-09-30/10-01).
# No logic of its own: the SAME per-dataset wrappers and settings the rung-5 runs used
# (each dataset's 06_evaluate/06_06_run_all_eval.sh: duke METRICS_SUBDIR=ablations/<item>,
# ispy1/acrin6698 LADDER=1), which sit on the shared 00_02_predict/predict_common.sh (cross mode)
# and 00_03_evaluate layer. Wrappers block until their run_job jobs finish (--wait).
#
#   bash scripts/cluster/vulcan_rung6_pv_breast_cross.sh fetch   # pull the 2 rung-6 model dirs from TamIA (verified)
#   bash scripts/cluster/vulcan_rung6_pv_breast_cross.sh run     # predict + eval, all 3 datasets x 2 models
# Run table: scripts/cluster/tamia_rung6_pv_runs.sh (RUN_IDs).
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
BC=benchmark/02_tasks/breast_cancer
MODE="${1:?usage: fetch|run}"
# tc | rung-6 RUN_ID | ispy2 Dataset dir | predict wrapper number (the rung-5 ladder wrappers, RUN_ID as $1)
MODELS=(
  "t1wce|ispy2_t1wce_v26_6_2_pv_train050_val100_20261003_111216|Dataset100_ISPY2T1wce|05_17_predict_ispy2_t1wce_v26_6_2_train050_val100.sh"
  "t2w|ispy2_t2w_v26_6_2_pv_train050_val100_20261003_111217|Dataset101_ISPY2T2w|05_21_predict_ispy2_t2w_v26_6_2_train050_val100.sh"
)
VULCAN_PRED="${BC}/ispy2/8_results_ispy2/01_predictions/ispy2_model"
TAMIA_PRED=/scratch/p/paulh/ispy2/8_results/01_predictions/ispy2_model

if [ "${MODE}" = fetch ]; then
  for m in "${MODELS[@]}"; do IFS='|' read -r tc rid ds _w <<<"$m"
    dst="${VULCAN_PRED}/${tc}/nnUNet/${rid}"; mkdir -p "${dst}"
    # model only (plans/dataset json + fold checkpoints/logs); no wandb, no prediction dirs
    ssh tamia.alliancecan.ca "cd ${TAMIA_PRED}/${tc}/nnUNet/${rid} && tar cf - --exclude=wandb ${ds}" | tar xf - -C "${dst}"
    for k in 0 1 2; do for ck in checkpoint_best checkpoint_final; do
      f="${ds}/nnUNetTrainerISPY2AugLabValSynth__nnUNetPlans__3d_fullres/fold_${k}/${ck}.pth"
      a=$(md5sum < "${dst}/${f}" | cut -c1-32); b=$(ssh tamia.alliancecan.ca "md5sum < ${TAMIA_PRED}/${tc}/nnUNet/${rid}/${f}" | cut -c1-32)
      [ "$a" = "$b" ] || { echo "[fetch] MISMATCH ${rid} ${f}"; exit 1; }
    done; done
    echo "[fetch] ${rid}: 6 checkpoints verified -> ${dst}"
  done
  exit 0
fi

[ "${MODE}" = run ] || { echo "unknown mode ${MODE}"; exit 2; }
one() {   # <dataset> <tc> <rid> <wrapper>
  local d="$1" tc="$2" rid="$3" w="$4" S="${BC}/$1/5_scripts_$1"
  case "$d" in
    duke-breast-mri)
      PREDICT_TIME_OVERRIDE=01:00:00 bash "${S}/05_predict/${w}" "${rid}" all t1wce_uniap precontrast_uniap
      for item in t1wce_uniap precontrast_uniap; do
        DUKE_ITEM="${item}" METRICS_SUBDIR="ablations/${item}" bash "${S}/06_evaluate/06_01_evaluate_ispy2_run.sh" "${rid}" nnUNet "${tc}"
      done ;;
    ispy1|acrin6698)
      bash "${S}/05_predict/${w}" "${rid}"
      LADDER=1 bash "${S}/06_evaluate/06_01_evaluate_run.sh" "${rid}" nnUNet "${tc}" ;;
  esac
  echo "[breast-cross] DONE ${d} ${tc} ${rid}"
}
pids=()
for m in "${MODELS[@]}"; do IFS='|' read -r tc rid ds w <<<"$m"
  [ -f "${VULCAN_PRED}/${tc}/nnUNet/${rid}/${ds}/nnUNetTrainerISPY2AugLabValSynth__nnUNetPlans__3d_fullres/fold_2/checkpoint_best.pth" ] \
    || { echo "model ${rid} not fetched -- run '$0 fetch' first"; exit 1; }
  for d in duke-breast-mri ispy1 acrin6698; do
    ( one "$d" "$tc" "$rid" "$w" ) > "/project/aip-jcohen/paulh/mri_synthesis_project/${BC}/$d/8_results_$d/rung6pv_${tc}.log" 2>&1 &
    pids+=($!); sleep 10
  done
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[breast-cross] all done rc=${rc}"; exit ${rc}
