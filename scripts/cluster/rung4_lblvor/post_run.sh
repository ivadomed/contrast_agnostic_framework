#!/bin/bash
# Post-training controller for ONE new noise-fill rung-4 run (`*_lblvor_*`, scripts/cluster/rung4_lblvor), 2026-10-07.
# CPU run_job behind the run's 3 fold jobs (queue_post.sh): (1) refuse unless checkpoints exist for folds 0-2; (2) predict
# every test source with the setting's rung-4 predict wrapper (same trainer + category as the new run; chaos gets its
# generated *_lblvor predict wrapper since the new run uses the plain trainer); (3) evaluate with the dataset's own
# evaluator, metrics placed exactly where the OLD rung-4 metrics sit; (4) audit vs the OLD rung-4 run (same cases).
# Status: $SCRATCH/rung4_lblvor/post/<RUN_ID>.status.      bash scripts/cluster/rung4_lblvor/post_run.sh <RUN_ID>
set -uo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
RID="${1:?usage: post_run.sh <RUN_ID>}"
# bash reads a script lazily: execute a private snapshot so edits to this file never hit a RUNNING controller
# (jobs that have not started yet still pick up the latest version)
if [ -z "${R4_SNAPSHOT:-}" ]; then
  snap="${SCRATCH:?}/rung4_lblvor/post/${RID}.post_run.snapshot.sh"; mkdir -p "$(dirname "${snap}")"
  cp "$0" "${snap}" && export R4_SNAPSHOT=1 && exec bash "${snap}" "$@"
fi
ROOT="$PWD"; T=benchmark/02_tasks; HERE="${ROOT}/scripts/cluster/rung5_val000"   # audit.py lives there
OUTD="${SCRATCH:?}/rung4_lblvor/post"; mkdir -p "${OUTD}"; ST="${OUTD}/${RID}.status"
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

audit() {   # audit <old rung-4 metrics dir or glob> <new run metrics dir> <label>
  local old; old="$(ls -d $1 2>/dev/null | head -1)"
  [ -n "${old}" ] || { log "audit ${3}: no old rung-4 metrics matching $1 -- skipped"; return 0; }
  "${ROOT}/.venv/bin/python" "${HERE}/audit.py" "${old}" "$2" "$3" 2>&1 | tee -a "${ST}"
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
# ── Pelvis (own ct + mri; old rung 4 flat) ──────────────────────────────────────────────────────────────────────
totalseg-pelvic_ct_*|totalseg-pelvic_mri_*)
  P="${T}/pelvis_healthy/totalseg-pelvic/5_scripts_totalseg-pelvic"; M="${T}/pelvis_healthy/totalseg-pelvic/8_results_totalseg-pelvic/02_metrics/totalseg_pelvic_model"
  case "${RID}" in *_ct_*) tc=ct; W=05_11_predict_ct_ladder_baseline_kmeans_label_remap_voronoi.sh ;; *) tc=mri; W=05_22_predict_mri_ladder_baseline_kmeans_label_remap_voronoi.sh ;; esac
  predict bash "${P}/05_predict/${W}" "${RID}"
  ev "export TRAINING_CONTRAST=${tc} METRICS_SUBDIR=; bash ${P}/06_evaluate/06_01_evaluate_run.sh ${RID} ${CAT}"
  audit "${M}/${tc}/*_${tc}_baseline_kmeans_label_remap_voronoi_2026091*" "${M}/${tc}/${CAT}_${RID}" "pelvis_${tc}"
  ;;
# ── Glioma (own 4 contrasts) ────────────────────────────────────────────────────────────────────────────────────
brats2024-glioma_*)
  B="${T}/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma"; M="${T}/brain_tumor/brats2024-glioma/8_results_brats2024-glioma/02_metrics/brats2024_glioma_model"
  case "${RID}" in
    *_t1n_*) tc=t1n; envf=env.sh;     W=05_46_predict_t1n_baseline_kmeans_label_remap_voronoi.sh; DS=051; OLD=20260730_200711 ;;
    *_t2w_*) tc=t2w; envf=env_t2w.sh; W=05_49_predict_t2w_baseline_kmeans_label_remap_voronoi.sh; DS=052; OLD=20260805_020659 ;;
    *_t2f_*) tc=t2f; envf=env_t2f.sh; W=05_60_predict_t2f_baseline_kmeans_label_remap_voronoi.sh; DS=053; OLD=20260917_094019 ;;
    *_t1c_*) tc=t1c; envf=env_t1c.sh; W=05_80_predict_t1c_baseline_kmeans_label_remap_voronoi.sh; DS=054; OLD=20260921_140000 ;;
  esac
  predict env TRAINING_CONTRAST="${tc}" bash "${B}/05_predict/${W}" "${RID}"
  ev "export TRAINING_CONTRAST=${tc}; source ${B}/00_utils/${envf}; export TRAINING_CONTRAST=${tc} DATASET_ID=${DS} EVAL_INLINE=1 CATEGORY=${CAT} METRICS_SUBDIR=ablations
      for F in 0 1 2; do bash ${B}/06_evaluate/06_01_evaluate_run.sh ${RID} \${F} || exit 1; done"
  audit "${M}/${tc}/ablations/*brats2024-glioma_${tc}_baseline_kmeans_label_remap_voronoi_${OLD}" "${M}/${tc}/ablations/${CAT}_${RID}" "brats_${tc}"
  ;;
# ── Brain (on-harmony; checkpoint_best; RAS->native resample inside the predict job) ─────────────────────────────
on-harmony_*)
  O="${T}/brain_healthy/on-harmony/5_scripts_on-harmony"; M="${T}/brain_healthy/on-harmony/8_results_on-harmony/02_metrics/on_harmony_model"
  case "${RID}" in
    on-harmony_T1w_*)    tc=T1w;    envf=env.sh;     W=05_06_predict_t1w_auglab_default.sh;    OLD=20260801_191042 ;;
    on-harmony_T2w_*)    tc=T2w;    envf=env_t2w.sh; W=05_12_predict_t2w_auglab_default.sh;    OLD=20260804_195921 ;;
    on-harmony_dwi_ap_*) tc=dwi_ap; envf=env_dwi.sh; W=05_18_predict_dwi_ap_auglab_default.sh; OLD=20260921_203727 ;;
  esac
  predict bash "${O}/05_predict/${W}" "${RID}"
  ev "source ${O}/00_utils/${envf}; export METRICS_SUBDIR=ablations; for F in 0 1 2; do bash ${O}/06_evaluate/06_01_evaluate_testset.sh ${RID} \${F} || exit 1; done"
  audit "${M}/${tc}/ablations/*on-harmony_${tc}_baseline_kmeans_label_remap_voronoi_${OLD}" "${M}/${tc}/ablations/${CAT}_${RID}" "onh_${tc}"
  ;;
# ── MS (open-ms own 3 contrasts) ────────────────────────────────────────────────────────────────────────────────
open-ms_t1w_*|open-ms_flair_*)
  S="${T}/brain_ms/open-ms/5_scripts_open-ms"; M="${T}/brain_ms/open-ms/8_results_open-ms/02_metrics/open_ms_model"
  case "${RID}" in
    open-ms_t1w_*)   tc=t1w;   W=05_37_predict_t1w_baseline_kmeans_label_remap_voronoi_train050_val000.sh; E=06_13_evaluate_t1w.sh; OLD=20260805_020341 ;;
    open-ms_flair_*) tc=flair; W=05_23_predict_baseline_kmeans_label_remap_voronoi_train050_val000.sh;     E=06_01_evaluate_run.sh;  OLD=20260711_062857 ;;
  esac
  predict bash "${S}/05_predict/${W}" "${RID}"
  ev "export METRICS_SUBDIR=ablations; bash ${S}/06_evaluate/${E} ${RID} ${CAT}"
  audit "${M}/${tc}/ablations/*open-ms_${tc}_baseline_kmeans_label_remap_voronoi_train050_val000_${OLD}" "${M}/${tc}/ablations/${CAT}_${RID}" "openms_${tc}"
  ;;
# ── Abdomen (chaos own + amos/sliver07 crop-before-predict; t1in old rung 4 flat, t2spir in ablations/) ───────────
chaos_t1in_*|chaos_t2spir_*)
  C="${T}/abdomen_healthy/chaos/5_scripts_chaos"; A="${T}/abdomen_healthy"; CM="${A}/chaos/8_results_chaos/02_metrics/chaos_model"
  case "${RID}" in
    chaos_t1in_*)   tc=t1in;   envf=env.sh;         DS=60; W=05_51_predict_t1in_baseline_kmeans_label_remap_voronoi_lblvor.sh;   SUB= ;
                    OLDG="${CM}/t1in/*chaos_t1in_baseline_kmeans_label_remap_voronoi_train050_val000_20260715_081959" ;;
    chaos_t2spir_*) tc=t2spir; envf=env_t2spir.sh;  DS=61; W=05_52_predict_t2spir_baseline_kmeans_label_remap_voronoi_lblvor.sh; SUB=ablations ;
                    OLDG="${CM}/t2spir/ablations/*baseline_kmeans_label_remap_voronoi_train050_val000*" ;;
  esac
  predict bash "${C}/05_predict/${W}" "${RID}"
  ev "source ${C}/00_utils/${envf}; export DATASET_ID=${DS} CATEGORY=${CAT} METRICS_SUBDIR=${SUB}; bash ${C}/06_evaluate/06_01_evaluate_run.sh ${RID}"
  audit "${OLDG}" "${CM}/${tc}${SUB:+/${SUB}}/${CAT}_${RID}" "chaos_${tc}_own"
  # external cohorts: shared crop-before-predict driver on the Vulcan-regenerated production crops (voxel-identical to TamIA's),
  # through the rung-5 mirror whose 8_results_* are symlinks into this checkout -> metrics land in the repo's fov_crop/.
  MIR="${SCRATCH}/rung5_val000/fovmirror"; CROPS="${SCRATCH}/srcsm_match/fovmirror"; RF="${OUTD}/${RID}.fovruns.txt"
  echo "${tc}|${CAT}|${RID}|$(basename "$(ls -d "${RDIR}"/Dataset*/*__nnUNetPlans__3d_fullres | head -1)" | sed 's/__nnUNetPlans__3d_fullres//')|${SUB}" > "${RF}"
  DRV="${ROOT}/benchmark/00_commun_scripts/00_02_predict/fov_crop_predict_evaluate.sh"
  for ds in amos sliver07; do
    if [ "$ds" = amos ]; then X="DATASET=amos ITEMS='ct mri' ANCHOR=kidney ANCHOR_IDS=2,3 EVAL_MODE=amos"
    else X="DATASET=sliver07 ITEMS=ct ANCHOR=liver ANCHOR_IDS=1 EVAL_MODE=generic_labels EVAL_LABELS=liver"; fi
    X="${X} CROP_REUSE_DIR=${CROPS}/${ds}/_fovcrop/vulcanprep"
    ( source scripts/job_runner/run_job.sh
      run_job --name "r4fov_${ds}_${RID: -15}" --gpus 1 --cpus 16 --mem 64G --time 03:00:00 --log "${OUTD}/${RID}.fov_${ds}_predict.log" --wait -- \
        bash -c "export SCRATCH=${MIR} RUNS_FILE=${RF} CONTRASTS=${tc} SKIP_REPORT=1 PHASES=2 FOV_NGPU=1 ${X}; bash ${DRV}" ) || fail "fov predict ${ds}"
    ( source scripts/job_runner/run_job.sh
      run_job --name "r4fovev_${ds}_${RID: -15}" --gpus 0 --cpus 16 --mem 120G --time 02:00:00 --log "${OUTD}/${RID}.fov_${ds}_eval.log" --wait -- \
        bash -c "export SCRATCH=${MIR} RUNS_FILE=${RF} CONTRASTS=${tc} SKIP_REPORT=1 PHASES='3 4' EVAL_PARALLEL=6 ${X}; bash ${DRV}" ) || fail "fov eval ${ds}"
    FM="${A}/${ds}/8_results_${ds}/02_metrics/chaos_model/${tc}/fov_crop"
    audit "${FM}/${SUB:+${SUB}/}*_baseline_kmeans_label_remap_voronoi_train050_val000_2026*" "${FM}/${SUB:+${SUB}/}${CAT}_${RID}" "chaos_${tc}_${ds}_fovcrop"
  done
  ;;
# ── Mandible (toothfairy2 own cbct + hanseg CT/MR + pddca CT on the S-I flipped _sif inputs; mandible-only) ─────
toothfairy2_cbct_*)
  MD="${ROOT}/${T}/mandible_healthy"; OLD="*toothfairy2_cbct_baseline_kmeans_label_remap_voronoi_20260908_013028"
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
  OLD="*ispy2_${tc}_baseline_kmeans_label_remap_voronoi_20260905_163655"
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
# ── Pancreas (pansegdata own + companions totalsegmri/amos/msd-pancreas; pinned AugLab like its other rungs) ────────
pansegdata_*)
  export CE_EXTRA_PYTHONPATH=/project/aip-jcohen/paulh/pansegdata_auglab_7b761b5/AugLab
  PZ="${T}/pancreas_disease"; S="${PZ}/pansegdata/5_scripts_pansegdata"; M="${PZ}/pansegdata/8_results_pansegdata/02_metrics/pansegdata_model"
  case "${RID}" in
    pansegdata_t1wce_*) tc=t1wce; envf=env.sh;     DS=150; W=05_11_predict_t1wce_baseline_kmeans_label_remap_voronoi.sh; OLD=20261005_133408 ;;
    pansegdata_t2w_*)   tc=t2w;   envf=env_t2w.sh; DS=151; W=05_22_predict_t2w_baseline_kmeans_label_remap_voronoi.sh;   OLD=20261005_133621 ;;
  esac
  TR="$(basename "$(ls -d "${RDIR}"/Dataset*/*__nnUNetPlans__3d_fullres | head -1)" | sed 's/__nnUNetPlans__3d_fullres//')"
  predict bash "${S}/05_predict/${W}" "${RID}"
  ev "source ${S}/00_utils/${envf}; export TRAINING_CONTRAST=${tc} METRICS_SUBDIR=ablations; bash ${S}/06_evaluate/06_01_evaluate_run.sh ${RID} ${CAT}"
  audit "${M}/${tc}/ablations/*pansegdata_${tc}_baseline_kmeans_label_remap_voronoi_${OLD}" "${M}/${tc}/ablations/${CAT}_${RID}" "pansegdata_${tc}_own"
  for comp in totalsegmri-pancreas amos-pancreas msd-pancreas; do
    CS="${PZ}/${comp}/5_scripts_${comp}"
    ( export PANSEG_TRAINING_CONTRAST=${tc} PANSEG_DATASET_ID=${DS} METHOD=baseline_kmeans_label_remap_voronoi_lblvor TRAINER=${TR} CATEGORY=${CAT}
      bash "${CS}/05_predict/05_01_predict_pansegdata_common.sh" "${RID}" all ) > "${OUTD}/${RID}.predict_${comp}.log" 2>&1 || { fail "predict ${comp}"; continue; }
    ev "export LADDER=1 EVAL_MEM=64G EVAL_TIME=3:00:00; bash ${CS}/06_evaluate/06_01_evaluate_run.sh ${RID} ${CAT} ${tc}"
    CM="${PZ}/${comp}/8_results_${comp}/02_metrics/pansegdata_model/${tc}"
    for nd in $(ls -d "${CM}"/ablations/*/"${CAT}_${RID}" 2>/dev/null); do
      it="$(basename "$(dirname "${nd}")")"
      audit "${CM}/ablations/${it}/*pansegdata_${tc}_baseline_kmeans_label_remap_voronoi_${OLD}" "${nd}" "${comp}_${tc}_${it}"
    done
  done
  ;;
*) log "FAILED: no recipe for ${RID}"; exit 3 ;;
esac

if [ "${FAIL}" = 0 ]; then log "DONE ${RID}"; else log "FAILED ${RID} (see above)"; fi
exit ${FAIL}
