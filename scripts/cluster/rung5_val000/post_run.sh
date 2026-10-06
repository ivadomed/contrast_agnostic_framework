#!/bin/bash
# Post-training controller for ONE val000 real-fill run (rung 5 "PALETTE alone" or rung-6 "PALETTE + PV alone"),
# 2026-10-06. Runs on Vulcan as a CPU run_job behind the run's 3 fold jobs (queued by queue_post.sh):
#   (1) refuse unless checkpoint_best + checkpoint_final exist for folds 0-2;
#   (2) predict every test source with the setting's RUNG-4 predict wrapper (same trainer class + category as the
#       new run; the predict wrappers submit their own GPU jobs and block with --wait);
#   (3) evaluate with the dataset's own evaluate script (RUN_JOB_INLINE=1: inside this job), metrics placed exactly
#       where the OLD rung-5 run's metrics sit for that source (ablations/ or flat), prefixed with the new category;
#   (4) audit every source against the old rung-5 run's eval_all.csv (same case set per fold/group; Dice side by side).
# Status: $SCRATCH/rung5_val000/post/<RUN_ID>.status (last line DONE / FAILED ...).
#   bash scripts/cluster/rung5_val000/post_run.sh <RUN_ID>
set -uo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
RID="${1:?usage: post_run.sh <RUN_ID>}"
# bash reads a script lazily: execute a private snapshot so edits to this file never hit a RUNNING controller
# (jobs that have not started yet still pick up the latest version)
if [ -z "${R5_SNAPSHOT:-}" ]; then
  snap="${SCRATCH:?}/rung5_val000/post/${RID}.post_run.snapshot.sh"; mkdir -p "$(dirname "${snap}")"
  cp "$0" "${snap}" && export R5_SNAPSHOT=1 && exec bash "${snap}" "$@"
fi
ROOT="$PWD"; T=benchmark/02_tasks; HERE="${ROOT}/scripts/cluster/rung5_val000"
OUTD="${SCRATCH:?}/rung5_val000/post"; mkdir -p "${OUTD}"; ST="${OUTD}/${RID}.status"
export RUN_JOB_ACCOUNT=aip-jcohen RUN_JOB_GPU_TYPE=l40s
# rack02-06: every GPU job there dies at CUDA init ("CUDA unknown error") yet ends COMPLETED -- silently missing predictions (2026-10-06)
export RUN_JOB_EXCLUDE_NODES="${RUN_JOB_EXCLUDE_NODES:-rack02-06}"
unset RUN_JOB_DEPENDENCY RUN_JOB_INLINE
log() { echo "[$(date '+%F %T')] $*" | tee -a "${ST}"; }
FAIL=0; fail() { log "ERROR: $*"; FAIL=1; }
log "START ${RID} host=$(hostname) job=${SLURM_JOB_ID:-none}"

# the TRAINING dir (holds Dataset*/<trainer>/fold_0), not a cross-dataset prediction dir of the same RUN_ID
# (amos/duke/acrin6698/hanseg... sort BEFORE the training dataset)
RDIR=""; for c in ${T}/*/*/8_results_*/01_predictions/*/*/*/"${RID}"; do
  ls -d "$c"/Dataset*/*/fold_0 >/dev/null 2>&1 && { RDIR="$c"; break; }; done
[ -n "${RDIR}" ] || { log "FAILED: no run dir for ${RID}"; exit 1; }
CAT="$(basename "$(dirname "${RDIR}")")"          # nnUNet | auglab
log "run dir ${RDIR} (category ${CAT})"

# (1) checkpoints
for k in 0 1 2; do for ck in checkpoint_best checkpoint_final; do
  ls "${RDIR}"/Dataset*/*/fold_${k}/${ck}.pth >/dev/null 2>&1 || { log "FAILED: missing ${ck} fold ${k} -- training incomplete; resume with the same RUN_ID, then re-run this script"; exit 2; }
done; done
log "checkpoints OK (best + final, folds 0-2)"

audit() {   # audit <old run metrics dir> <new run metrics dir> <label>
  "${ROOT}/.venv/bin/python" "${HERE}/audit.py" "$1" "$2" "$3" 2>&1 | tee -a "${ST}"
  [ "${PIPESTATUS[0]}" = 0 ] || FAIL=1
}
inline_eval() { ( export RUN_JOB_INLINE=1; "$@" ) 2>&1 | tail -5 | tee -a "${ST}"; [ "${PIPESTATUS[0]}" = 0 ] || fail "eval: $*"; }
predict() {   # log named after the dataset + wrapper (several predicts per run)
  local w; w="$(printf '%s\n' "$@" | grep -m1 '\.sh$')"; local tag; tag="$(basename "$(dirname "$(dirname "$w")")")_$(basename "$w" .sh)"
  "$@" > "${OUTD}/${RID}.predict_${tag}.log" 2>&1 || fail "predict: $*"; }

# ---------------------------------------------------------------------------------------------------------------
PV=0; case "${RID}" in *_v26_6_2_pv_train050_val000_*) PV=1 ;; esac
# metrics placement: real fill next to its rung 4; PV next to its rung-6 registry key (ladder_pv_branch.yaml)
sub_for() { if [ "${PV}" = 1 ]; then echo "$2"; else echo "$1"; fi; }    # sub_for <real> <pv>
ev() { inline_eval bash -c "$1"; }                                 # ev '<bash snippet>' -- runs with RUN_JOB_INLINE=1
case "${RID}" in
# ── Pelvis (own ct + mri) ───────────────────────────────────────────────────────────────────────────────────────
totalseg-pelvic_ct_*|totalseg-pelvic_mri_*)
  P="${T}/pelvis_healthy/totalseg-pelvic/5_scripts_totalseg-pelvic"
  M="${T}/pelvis_healthy/totalseg-pelvic/8_results_totalseg-pelvic/02_metrics/totalseg_pelvic_model"
  case "${RID}" in
    *_ct_*)  tc=ct;  W=05_11_predict_ct_ladder_baseline_kmeans_label_remap_voronoi.sh;  OLD=nnUNet_ct_v26_6_2_train050_val100_20260916_072434 ;;
    *_mri_*) tc=mri; W=05_22_predict_mri_ladder_baseline_kmeans_label_remap_voronoi.sh; OLD=nnUNet_mri_v26_6_2_train050_val100_20260916_072453 ;;
  esac
  SUB="$(sub_for "" ablations)"
  predict bash "${P}/05_predict/${W}" "${RID}"
  ev "export TRAINING_CONTRAST=${tc} METRICS_SUBDIR=${SUB}; bash ${P}/06_evaluate/06_01_evaluate_run.sh ${RID} ${CAT}"
  audit "${M}/${tc}/${OLD}" "${M}/${tc}${SUB:+/${SUB}}/${CAT}_${RID}" "pelvis_${tc}"
  ;;
# ── Glioma (own 4 contrasts; GT = labelsTr of the training Dataset, as for every brats run) ────────────────────
brats2024-glioma_*)
  B="${T}/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma"
  M="${T}/brain_tumor/brats2024-glioma/8_results_brats2024-glioma/02_metrics/brats2024_glioma_model"
  case "${RID}" in
    *_t1n_*) tc=t1n; envf=env.sh;     W=05_46_predict_t1n_baseline_kmeans_label_remap_voronoi.sh; DS=051; OLD=nnUNet_brats2024-glioma_t1n_v26_6_2_train050_val100_20260730_200711 ;;
    *_t2w_*) tc=t2w; envf=env_t2w.sh; W=05_49_predict_t2w_baseline_kmeans_label_remap_voronoi.sh; DS=052; OLD=nnUNet_brats2024-glioma_t2w_v26_6_2_train050_val100_20261003_123018 ;;
    *_t2f_*) tc=t2f; envf=env_t2f.sh; W=05_60_predict_t2f_baseline_kmeans_label_remap_voronoi.sh; DS=053; OLD=nnUNet_brats2024-glioma_t2f_v26_6_2_train050_val100_20260917_113412 ;;
    *_t1c_*) tc=t1c; envf=env_t1c.sh; W=05_80_predict_t1c_baseline_kmeans_label_remap_voronoi.sh; DS=054; OLD=nnUNet_brats2024-glioma_t1c_v26_6_2_train050_val100_20260921_140000 ;;
  esac
  predict env TRAINING_CONTRAST="${tc}" bash "${B}/05_predict/${W}" "${RID}"
  # folds sequential, in-process (3 folds x 4 contrasts at once OOM-killed a 64G job before)
  ev "export TRAINING_CONTRAST=${tc}; source ${B}/00_utils/${envf}; export TRAINING_CONTRAST=${tc} DATASET_ID=${DS} EVAL_INLINE=1 CATEGORY=${CAT} METRICS_SUBDIR=ablations
      for F in 0 1 2; do bash ${B}/06_evaluate/06_01_evaluate_run.sh ${RID} \${F} || exit 1; done"
  audit "${M}/${tc}/ablations/${OLD}" "${M}/${tc}/ablations/${CAT}_${RID}" "brats_${tc}"
  ;;
# ── Brain (on-harmony; checkpoint_final by default for predict AND eval, as for all its runs) ──────────────────
on-harmony_*)
  O="${T}/brain_healthy/on-harmony/5_scripts_on-harmony"
  M="${T}/brain_healthy/on-harmony/8_results_on-harmony/02_metrics/on_harmony_model"
  case "${RID}" in
    on-harmony_T1w_*)    tc=T1w;    envf=env.sh;     W=05_06_predict_t1w_auglab_default.sh;    OLD=ablations/nnUNet_on-harmony_T1w_v26_6_2_train050_val100_20261003_123018 ;;
    on-harmony_T2w_*)    tc=T2w;    envf=env_t2w.sh; W=05_12_predict_t2w_auglab_default.sh;    OLD=nnUNet_on-harmony_T2w_v26_6_2_train050_val100_20260625_154418 ;;
    on-harmony_dwi_ap_*) tc=dwi_ap; envf=env_dwi.sh; W=05_18_predict_dwi_ap_auglab_default.sh; OLD=nnUNet_on-harmony_dwi_ap_v26_6_2_train050_val100_20260921_203727 ;;
  esac
  SUB="$(sub_for ablations "")"
  # predict_common blocks per fold (run_job --wait), so the shim's RAS->native resample runs AFTER the predictions exist
  predict bash "${O}/05_predict/${W}" "${RID}"
  ev "source ${O}/00_utils/${envf}; export METRICS_SUBDIR=${SUB}; for F in 0 1 2; do bash ${O}/06_evaluate/06_01_evaluate_testset.sh ${RID} \${F} || exit 1; done"
  audit "${M}/${tc}/${OLD}" "${M}/${tc}${SUB:+/${SUB}}/${CAT}_${RID}" "onh_${tc}"
  ;;
# ── MS T1w (open-ms own 3 contrasts) ────────────────────────────────────────────────────────────────────────────
open-ms_t1w_*)
  S="${T}/brain_ms/open-ms/5_scripts_open-ms"; M="${T}/brain_ms/open-ms/8_results_open-ms/02_metrics/open_ms_model"
  predict bash "${S}/05_predict/05_37_predict_t1w_baseline_kmeans_label_remap_voronoi_train050_val000.sh" "${RID}"
  ev "export METRICS_SUBDIR=ablations; bash ${S}/06_evaluate/06_13_evaluate_t1w.sh ${RID} ${CAT}"
  audit "${M}/t1w/nnUNet_open-ms_t1w_v26_6_2_train050_val100_20260708_083641" "${M}/t1w/ablations/${CAT}_${RID}" "openms_t1w"
  ;;
# ── Abdomen T2spir (chaos own + amos/sliver07 crop-before-predict) ─────────────────────────────────────────────
chaos_t2spir_*)
  C="${T}/abdomen_healthy/chaos/5_scripts_chaos"; A="${T}/abdomen_healthy"
  OLD=nnUNet_chaos_t2spir_v26_6_2_train050_val100_20261003_123018
  predict bash "${C}/05_predict/05_47_predict_t2spir_baseline_kmeans_label_remap_voronoi.sh" "${RID}"
  ev "source ${C}/00_utils/env_t2spir.sh; export DATASET_ID=61 CATEGORY=${CAT} METRICS_SUBDIR=ablations; bash ${C}/06_evaluate/06_01_evaluate_run.sh ${RID}"
  audit "${A}/chaos/8_results_chaos/02_metrics/chaos_model/t2spir/ablations/${OLD}" "${A}/chaos/8_results_chaos/02_metrics/chaos_model/t2spir/ablations/${CAT}_${RID}" "chaos_t2spir_own"
  # cross-dataset: the shared crop-before-predict driver (06_36 recipe), on the EXACT TamIA crops (copied) through a
  # $SCRATCH mirror of symlinks into this checkout -- predictions + metrics land in the repo's amos/sliver07 trees.
  MIR="${SCRATCH}/rung5_val000/fovmirror"; RF="${OUTD}/${RID}.fovruns.txt"
  echo "t2spir|${CAT}|${RID}|$(basename "$(ls -d "${RDIR}"/Dataset*/*__nnUNetPlans__3d_fullres | head -1)" | sed 's/__nnUNetPlans__3d_fullres//')|ablations" > "${RF}"
  DRV="${ROOT}/benchmark/00_commun_scripts/00_02_predict/fov_crop_predict_evaluate.sh"
  for ds in amos sliver07; do
    if [ "$ds" = amos ]; then X="DATASET=amos ITEMS='ct mri' ANCHOR=kidney ANCHOR_IDS=2,3 EVAL_MODE=amos CROP_REUSE_DIR=${MIR}/amos/_fovcrop/391947"
    else X="DATASET=sliver07 ITEMS=ct ANCHOR=liver ANCHOR_IDS=1 EVAL_MODE=generic_labels EVAL_LABELS=liver CROP_REUSE_DIR=${MIR}/sliver07/_fovcrop/391949"; fi
    ( source scripts/job_runner/run_job.sh
      run_job --name "r5fov_${ds}_${RID: -15}" --gpus 4 --cpus 32 --mem 160G --time 02:00:00 --log "${OUTD}/${RID}.fov_${ds}_predict.log" --wait -- \
        bash -c "export SCRATCH=${MIR} RUNS_FILE=${RF} CONTRASTS=t2spir SKIP_REPORT=1 PHASES=2 ${X}; bash ${DRV}" ) || fail "fov predict ${ds}"
    ( source scripts/job_runner/run_job.sh
      run_job --name "r5fovev_${ds}_${RID: -15}" --gpus 0 --cpus 16 --mem 120G --time 02:00:00 --log "${OUTD}/${RID}.fov_${ds}_eval.log" --wait -- \
        bash -c "export SCRATCH=${MIR} RUNS_FILE=${RF} CONTRASTS=t2spir SKIP_REPORT=1 PHASES='3 4' EVAL_PARALLEL=6 ${X}; bash ${DRV}" ) || fail "fov eval ${ds}"
    FM="${A}/${ds}/8_results_${ds}/02_metrics/chaos_model/t2spir/fov_crop/ablations"
    audit "${FM}/${OLD}" "${FM}/${CAT}_${RID}" "chaos_t2spir_${ds}_fovcrop"
  done
  ;;
# ── Mandible (toothfairy2 own cbct + hanseg CT/MR + pddca CT on the S-I flipped _sif inputs; mandible-only) ─────
toothfairy2_cbct_*)
  MD="${ROOT}/${T}/mandible_healthy"; OLD=nnUNet_toothfairy2_cbct_v26_6_2_train050_val100_20260908_013028
  predict bash "${MD}/toothfairy2/5_scripts_toothfairy2/05_predict/05_11_predict_baseline_kmeans_label_remap_voronoi.sh" "${RID}"
  predict env PREDICT_INPUT_SUFFIX=_sif PREDICT_OUTPUT_SUBDIR=sif bash "${MD}/hanseg/5_scripts_hanseg/05_predict/05_11_predict_baseline_kmeans_label_remap_voronoi.sh" "${RID}" all ct mrt1
  predict env PREDICT_INPUT_SUFFIX=_sif PREDICT_OUTPUT_SUBDIR=sif bash "${MD}/pddca/5_scripts_pddca/05_predict/05_11_predict_baseline_kmeans_label_remap_voronoi.sh" "${RID}" all ct
  TM="${MD}/toothfairy2/8_results_toothfairy2/02_metrics/toothfairy2_model/cbct"
  ev "export TF2_PRED=${MD}/toothfairy2/8_results_toothfairy2/01_predictions/toothfairy2_model/cbct TF2_METRICS=${TM} TF2_METRICS_SRC=${TM} \
        TF2_RAW=${MD}/toothfairy2/2_nnUNet_toothfairy2/raw/Dataset110_ToothFairy2CBCT FORCE_SUB=ablations ONLY_RUN=${RID}
      bash ${MD}/toothfairy2/5_scripts_toothfairy2/06_evaluate/06_08_evaluate_cbct.sh"
  HM="${MD}/hanseg/8_results_hanseg/02_metrics/toothfairy2_model/cbct"
  ev "export HS_PRED=${MD}/hanseg/8_results_hanseg/01_predictions/toothfairy2_model/cbct HS_METRICS=${HM} MO_DIR= FORCE_SUB=ablations \
        GT_OVERRIDE=${MD}/hanseg/2_nnUNet_hanseg/raw/labelsTs_ct_sif PRED_SUBDIR=sif ONLY_RUN=${RID}
      bash ${MD}/hanseg/5_scripts_hanseg/06_evaluate/06_03_eval_mandible_only.sh"
  PM="${MD}/pddca/8_results_pddca/02_metrics/toothfairy2_model/cbct"
  ev "export PD_PRED=${MD}/pddca/8_results_pddca/01_predictions/toothfairy2_model/cbct PD_METRICS=${PM} MO_DIR= \
        GT_OVERRIDE=${MD}/pddca/2_nnUNet_pddca/raw/labelsTs_ct_sif PRED_SUBDIR=sif ONLY_RUN=${RID}
      bash ${MD}/pddca/5_scripts_pddca/06_evaluate/06_01_eval_mandible_only.sh"
  audit "${TM}/ablations/${OLD}" "${TM}/ablations/${CAT}_${RID}" "tf2_cbct"
  audit "${HM}/ablations/${OLD}" "${HM}/ablations/${CAT}_${RID}" "hanseg_sif"
  audit "${PM}/ablations/${OLD}" "${PM}/ablations/${CAT}_${RID}" "pddca_sif"
  ;;
# ── Breast (ispy2 own + duke *_uniap / ispy1 / acrin6698 dwi_uniap) ─────────────────────────────────────────────
ispy2_t1wce_*|ispy2_t2w_*)
  BC="${T}/breast_cancer"; I="${BC}/ispy2/5_scripts_ispy2"
  case "${RID}" in
    ispy2_t1wce_*) tc=t1wce; envf=env.sh;     W=05_32_predict_own_t1wce_baseline_kmeans_label_remap_voronoi.sh; DS=100; XW=05_16_predict_ispy2_t1wce_baseline_kmeans_label_remap_voronoi.sh ;;
    ispy2_t2w_*)   tc=t2w;   envf=env_t2w.sh; W=05_47_predict_own_t2w_baseline_kmeans_label_remap_voronoi.sh;   DS=101; XW=05_20_predict_ispy2_t2w_baseline_kmeans_label_remap_voronoi.sh ;;
  esac
  OLD="nnUNet_ispy2_${tc}_v26_6_2_train050_val100_20260905_163655"
  predict bash "${I}/05_predict/${W}" "${RID}"
  predict env PREDICT_TIME_OVERRIDE=01:00:00 bash "${BC}/duke-breast-mri/5_scripts_duke-breast-mri/05_predict/${XW}" "${RID}" all t1wce_uniap precontrast_uniap
  predict bash "${BC}/ispy1/5_scripts_ispy1/05_predict/${XW}" "${RID}"
  predict bash "${BC}/acrin6698/5_scripts_acrin6698/05_predict/${XW}" "${RID}"
  ev "source ${I}/00_utils/${envf}; export METRICS_SUBDIR=ablations; bash ${I}/06_evaluate/06_01_evaluate_own_run.sh ${RID} ${CAT} ${DS} all"
  for item in t1wce_uniap precontrast_uniap; do
    ev "export DUKE_ITEM=${item} METRICS_SUBDIR=ablations/${item}; bash ${BC}/duke-breast-mri/5_scripts_duke-breast-mri/06_evaluate/06_01_evaluate_ispy2_run.sh ${RID} ${CAT} ${tc}"
  done
  ev "export LADDER=1; bash ${BC}/ispy1/5_scripts_ispy1/06_evaluate/06_01_evaluate_run.sh ${RID} ${CAT} ${tc}"
  ev "export LADDER=1; bash ${BC}/acrin6698/5_scripts_acrin6698/06_evaluate/06_01_evaluate_run.sh ${RID} ${CAT} ${tc}"
  audit "${BC}/ispy2/8_results_ispy2/02_metrics/ispy2_model/${tc}/ablations/${OLD}" "${BC}/ispy2/8_results_ispy2/02_metrics/ispy2_model/${tc}/ablations/${CAT}_${RID}" "ispy2_${tc}_own"
  for x in duke-breast-mri/t1wce_uniap duke-breast-mri/precontrast_uniap ispy1/t1wce ispy1/precontrast acrin6698/dwi_uniap; do
    xm="${BC}/${x%%/*}/8_results_${x%%/*}/02_metrics/ispy2_model/${tc}/ablations/${x#*/}"
    audit "${xm}/${OLD}" "${xm}/${CAT}_${RID}" "ispy2_${tc}_${x/\//_}"
  done
  ;;
*) log "FAILED: no recipe for ${RID}"; exit 3 ;;
esac

if [ "${FAIL}" = 0 ]; then log "DONE ${RID}"; else log "FAILED ${RID} (see above)"; fi
exit ${FAIL}
