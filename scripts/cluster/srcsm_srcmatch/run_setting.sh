#!/bin/bash
# SRCSM + its published test-time SOURCE MATCHING, for ONE training setting (2026-10-07; review item 5).
# CPU controller (submit via run_job, see launch.sh). For the setting's existing SRCSM model:
#   (1) build source-matched copies of every test input it is scored on (srcsm_match_setting.sh: average CDF of the
#       training images, histogram-match each test image, QC, symlink <input>_srcmatch_<tc> next to the original);
#   (2) predict with the setting's own SRCSM predict wrapper(s) on those inputs
#       (PREDICT_INPUT_SUFFIX=_srcmatch_<tc>, PREDICT_OUTPUT_SUBDIR=srcmatch -> fold{k}/srcmatch/<item>);
#   (3) evaluate with each dataset's own evaluator and CKPT_TAG=srcmatch -> metrics <cat>_<RUN_ID>_srcmatch next to
#       the plain SRCSM metrics (also inside fov_crop/ for the Abdomen external cohorts);
#   (4) audit: same cases per fold/group as the plain SRCSM run (scripts/cluster/rung5_val000/audit.py).
# The plain SRCSM metrics are never touched. Status: $SCRATCH/srcsm_match/_status/<key>.status
#   bash scripts/cluster/srcsm_srcmatch/run_setting.sh <key>
#   keys: ms_flair ms_t1w glioma_t1n glioma_t2w glioma_t1c glioma_t2f abdomen_t1in abdomen_t2spir brain_t1w brain_t2w brain_dwi
#         breast_t1wce breast_t2w mandible pelvis_ct pelvis_mri
set -uo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
KEY="${1:?usage: run_setting.sh <key>}"
if [ -z "${SM_SNAPSHOT:-}" ]; then     # run a private copy so later edits never hit a running controller
  snap="${SCRATCH:?}/srcsm_match/_status/${KEY}.snapshot.sh"; mkdir -p "$(dirname "${snap}")"
  cp "$0" "${snap}" && export SM_SNAPSHOT=1 && exec bash "${snap}" "$@"
fi
ROOT="$PWD"; T="${ROOT}/benchmark/02_tasks"
OUTD="${SCRATCH}/srcsm_match/_status"; ST="${OUTD}/${KEY}.status"
export RUN_JOB_ACCOUNT=aip-jcohen RUN_JOB_GPU_TYPE=l40s RUN_JOB_EXCLUDE_NODES="${RUN_JOB_EXCLUDE_NODES:-rack02-06}"
unset RUN_JOB_DEPENDENCY RUN_JOB_INLINE
MATCH="${ROOT}/benchmark/00_commun_scripts/00_02_predict/srcsm_match_setting.sh"
AUDIT="${ROOT}/scripts/cluster/rung5_val000/audit.py"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${ST}"; }
FAIL=0; fail() { log "ERROR: $*"; FAIL=1; }
log "START ${KEY} host=$(hostname) job=${SLURM_JOB_ID:-none}"
match() { ( export RUN_JOB_INLINE=1; bash "${MATCH}" "$@" ) > "${OUTD}/${KEY}.match.log" 2>&1 || { fail "match (see ${KEY}.match.log)"; return 1; }
          grep -E "KS before|excluded|average CDF" "${OUTD}/${KEY}.match.log" | tee -a "${ST}"; }
predict() { local w; w="$(printf '%s\n' "$@" | grep -m1 '\.sh$')"
            env PREDICT_INPUT_SUFFIX="_srcmatch_${TC}" PREDICT_OUTPUT_SUBDIR=srcmatch "$@" > "${OUTD}/${KEY}.predict_$(basename "$(dirname "$(dirname "$w")")")_$(basename "$w" .sh).log" 2>&1 || fail "predict: $*"; }
ev() { ( export RUN_JOB_INLINE=1 CKPT_TAG=srcmatch; bash -c "$1" ) 2>&1 | tail -4 | tee -a "${ST}"; [ "${PIPESTATUS[0]}" = 0 ] || fail "eval: $1"; }
mkexcl() { local o="${OUTD}/${KEY}.exclude.json"   # held-out cases that sit inside imagesTr: the case ids of these test dirs
           ls "$@" | grep '_0000.nii.gz$' | sed 's/_0000.nii.gz$//' | sort -u | "${ROOT}/.venv/bin/python" -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))" > "$o"; echo "$o"; }
audit() { "${ROOT}/.venv/bin/python" "${AUDIT}" "$1" "$2" "$3" 2>&1 | tee -a "${ST}"; [ "${PIPESTATUS[0]}" = 0 ] || FAIL=1; }

case "${KEY}" in
# ── MS (open-ms own 3 contrasts) ────────────────────────────────────────────────────────────────────────────────
ms_flair|ms_t1w)
  S="${T}/brain_ms/open-ms/5_scripts_open-ms"; M="${T}/brain_ms/open-ms/8_results_open-ms/02_metrics/open_ms_model"
  if [ "${KEY}" = ms_flair ]; then TC=flair; R="${T}/brain_ms/open-ms/2_nnUNet_open-ms/raw/Dataset070_OpenMS_FLAIR"; RID=open-ms_flair_srcsm_20260709_072043; W=05_18_predict_srcsm.sh; E=06_01_evaluate_run.sh
  else TC=t1w; R="${T}/brain_ms/open-ms/2_nnUNet_open-ms/raw/Dataset071_OpenMS_T1W"; RID=open-ms_t1w_srcsm_20260709_075121; W=05_19_predict_t1w_srcsm.sh; E=06_13_evaluate_t1w.sh; fi
  match "ms_${TC}" "${TC}" "${R}/imagesTr" mr "${R}/imagesTs_flair:mr" "${R}/imagesTs_t1w:mr" "${R}/imagesTs_t2w:mr" || exit 1
  predict bash "${S}/05_predict/${W}" "${RID}" all
  ev "bash ${S}/06_evaluate/${E} ${RID} auglab"
  audit "${M}/${TC}/auglab_${RID}" "${M}/${TC}/auglab_${RID}_srcmatch" "ms_${TC}"
  ;;
# ── Glioma (own 4 contrasts; the 70 test cases sit in imagesTr -> excluded from the source CDF) ────────────────
glioma_t1n|glioma_t2w|glioma_t1c|glioma_t2f)
  B="${T}/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma"; M="${T}/brain_tumor/brats2024-glioma/8_results_brats2024-glioma/02_metrics/brats2024_glioma_model"
  RAW="${T}/brain_tumor/brats2024-glioma/2_nnUNet_brats2024-glioma/raw"
  if [ "${KEY}" = glioma_t1n ]; then TC=t1n; envf=env.sh; DS=051; R="${RAW}/Dataset051_BraTS2024GliomaT1n"; RID=brats2024-glioma_t1n_srcsm_20260709_122015; W=05_18_predict_t1n_srcsm.sh
  elif [ "${KEY}" = glioma_t2w ]; then TC=t2w; envf=env_t2w.sh; DS=052; R="${RAW}/Dataset052_BraTS2024GliomaT2w"; RID=brats2024-glioma_t2w_srcsm_20260709_122045; W=05_19_predict_t2w_srcsm.sh
  elif [ "${KEY}" = glioma_t2f ]; then TC=t2f; envf=env_t2f.sh; DS=053; R="${RAW}/Dataset053_BraTS2024GliomaT2f"; RID=brats2024-glioma_t2f_srcsm_20260917_094019; W=05_54_predict_t2f_srcsm.sh
  else TC=t1c; envf=env_t1c.sh; DS=054; R="${RAW}/Dataset054_BraTS2024GliomaT1c"; RID=brats2024-glioma_t1c_srcsm_20260921_140000; W=05_74_predict_t1c_srcsm.sh; fi   # t1c/t2f models copied from TamIA
  EXCLUDE_CASES="${T}/brain_tumor/brats2024-glioma/4_splits_brats2024-glioma/test_cases.json" \
    match "glioma_${TC}" "${TC}" "${R}/imagesTr" mr "${R}/imagesTs_t1n:mr" "${R}/imagesTs_t1c:mr" "${R}/imagesTs_t2w:mr" "${R}/imagesTs_t2f:mr" || exit 1
  predict env TRAINING_CONTRAST="${TC}" bash "${B}/05_predict/${W}" "${RID}" all
  ev "export TRAINING_CONTRAST=${TC}; source ${B}/00_utils/${envf}; export TRAINING_CONTRAST=${TC} DATASET_ID=${DS} EVAL_INLINE=1 CATEGORY=auglab
      for F in 0 1 2; do bash ${B}/06_evaluate/06_01_evaluate_run.sh ${RID} \${F} || exit 1; done"
  audit "${M}/${TC}/auglab_${RID}" "${M}/${TC}/auglab_${RID}_srcmatch" "glioma_${TC}"
  ;;
# ── Abdomen (chaos own 4 items + amos/sliver07 crop-before-predict on the EXACT production crops) ───────────────
abdomen_t1in|abdomen_t2spir)
  C="${T}/abdomen_healthy/chaos/5_scripts_chaos"; A="${T}/abdomen_healthy"
  if [ "${KEY}" = abdomen_t1in ]; then TC=t1in; envf=env.sh; DS=60; R="${A}/chaos/2_nnUNet_chaos/raw/Dataset060_CHAOS_MR_T1in"; RID=chaos_t1in_srcsm_20260710_011817; W=05_32_predict_srcsm.sh
  else TC=t2spir; envf=env_t2spir.sh; DS=61; R="${A}/chaos/2_nnUNet_chaos/raw/Dataset061_CHAOS_MR_T2spir"; RID=chaos_t2spir_srcsm_20260709_121945; W=05_33_predict_t2spir_srcsm.sh; fi
  # crops regenerated on Vulcan by the shared driver's PHASE 1 (voxel- and affine-identical to TamIA's production crops
  # 391947/391949 for t2spir, checked 2026-10-07; t1in's were never copied to Vulcan)
  MIR="${SCRATCH}/rung5_val000/fovmirror"; CROPS="${SCRATCH}/srcsm_match/fovmirror"
  CA="${CROPS}/amos/_fovcrop/vulcanprep/${TC}"; CS="${CROPS}/sliver07/_fovcrop/vulcanprep/${TC}"
  match "abdomen_${TC}" "${TC}" "${R}/imagesTr" mr "${R}/imagesTs_t1in:mr" "${R}/imagesTs_t1out:mr" "${R}/imagesTs_t2spir:mr" "${R}/imagesTs_ct:ct" \
        "${CA}/imagesTs_ct:ct" "${CA}/imagesTs_mri:mr" "${CS}/imagesTs_ct:ct" || exit 1
  predict bash "${C}/05_predict/${W}" "${RID}" all
  ev "source ${C}/00_utils/${envf}; export DATASET_ID=${DS} CATEGORY=auglab; bash ${C}/06_evaluate/06_01_evaluate_run.sh ${RID}"
  audit "${A}/chaos/8_results_chaos/02_metrics/chaos_model/${TC}/auglab_${RID}" "${A}/chaos/8_results_chaos/02_metrics/chaos_model/${TC}/auglab_${RID}_srcmatch" "chaos_${TC}_own"
  # external cohorts: a crop work dir whose imagesTs_<item> are the MATCHED crops (labels = the production crops), predicted +
  # evaluated by the shared driver into a scratch results root, metrics then copied to fov_crop/<cat>_<RUN_ID>_srcmatch in the repo
  DRV="${ROOT}/benchmark/00_commun_scripts/00_02_predict/fov_crop_predict_evaluate.sh"
  SMIR="${SCRATCH}/srcsm_match/fovmirror"; RF="${OUTD}/${KEY}.fovruns.txt"
  TR="$(basename "$(ls -d "${A}"/chaos/8_results_chaos/01_predictions/chaos_model/${TC}/auglab/${RID}/Dataset*/*__nnUNetPlans__3d_fullres | head -1)" | sed 's/__nnUNetPlans__3d_fullres//')"
  echo "${TC}|auglab|${RID}|${TR}|" > "${RF}"
  for ds in amos sliver07; do
    if [ "$ds" = amos ]; then items="ct mri"; X="DATASET=amos ITEMS='ct mri' ANCHOR=kidney ANCHOR_IDS=2,3 EVAL_MODE=amos"
    else items="ct"; X="DATASET=sliver07 ITEMS=ct ANCHOR=liver ANCHOR_IDS=1 EVAL_MODE=generic_labels EVAL_LABELS=liver"; fi
    W2="${SMIR}/${ds}/_fovcrop/srcmatch/${TC}"; mkdir -p "${W2}" "${SMIR}/${ds}/8_results_${ds}"
    for d in 2_nnUNet_${ds}; do [ -e "${SMIR}/${ds}/${d}" ] || ln -s "${MIR}/${ds}/${d}" "${SMIR}/${ds}/${d}"; done
    [ -e "${SMIR}/chaos" ] || ln -s "${MIR}/chaos" "${SMIR}/chaos"
    PROD="${CROPS}/${ds}/_fovcrop/vulcanprep/${TC}"
    for it in ${items}; do
      ln -sfn "${PROD}/imagesTs_${it}_srcmatch_${TC}" "${W2}/imagesTs_${it}"; ln -sfn "${PROD}/labelsTs_${it}" "${W2}/labelsTs_${it}"
    done
    ( source scripts/job_runner/run_job.sh
      run_job --name "smfov_${ds}_${TC}" --gpus 1 --cpus 16 --mem 64G --time 02:00:00 --log "${OUTD}/${KEY}.fov_${ds}_predict.log" --wait -- \
        bash -c "export SCRATCH=${SMIR} RUNS_FILE=${RF} CONTRASTS=${TC} SKIP_REPORT=1 PHASES=2 FOV_NGPU=1 CROP_REUSE_DIR=${SMIR}/${ds}/_fovcrop/srcmatch ${X}; bash ${DRV}" ) || fail "fov predict ${ds}"
    ( source scripts/job_runner/run_job.sh
      run_job --name "smfovev_${ds}_${TC}" --gpus 0 --cpus 16 --mem 120G --time 02:00:00 --log "${OUTD}/${KEY}.fov_${ds}_eval.log" --wait -- \
        bash -c "export SCRATCH=${SMIR} RUNS_FILE=${RF} CONTRASTS=${TC} SKIP_REPORT=1 PHASES='3 4' EVAL_PARALLEL=6 CROP_REUSE_DIR=${SMIR}/${ds}/_fovcrop/srcmatch ${X}; bash ${DRV}" ) || fail "fov eval ${ds}"
    SRC="${SMIR}/${ds}/8_results_${ds}/02_metrics/chaos_model/${TC}/fov_crop/auglab_${RID}"
    DST="${A}/${ds}/8_results_${ds}/02_metrics/chaos_model/${TC}/fov_crop/auglab_${RID}_srcmatch"
    if [ -d "${SRC}" ]; then mkdir -p "$(dirname "${DST}")"; rsync -a "${SRC}/" "${DST}/"; else fail "no fov metrics ${SRC}"; fi
    audit "${A}/${ds}/8_results_${ds}/02_metrics/chaos_model/${TC}/fov_crop/auglab_${RID}" "${DST}" "chaos_${TC}_${ds}_fovcrop"
  done
  ;;
# ── Brain (on-harmony; the RAS->native resample runs inside each predict job; evaluator tags via CHECKPOINT) ────
brain_t1w|brain_t2w|brain_dwi)
  O="${T}/brain_healthy/on-harmony/5_scripts_on-harmony"; M="${T}/brain_healthy/on-harmony/8_results_on-harmony/02_metrics/on_harmony_model"
  RAW="${T}/brain_healthy/on-harmony/2_nnUNet_on-harmony/raw"; IN="${RAW}/Dataset031_OnHarmonyT1w31"
  case "${KEY}" in
    brain_t1w) TC=T1w;    envf=env.sh;     R="${RAW}/Dataset031_OnHarmonyT1w31"; RID=on-harmony_T1w_srcsm_20260709_122115;    W=05_07_predict_t1w_srcsm.sh ;;
    brain_t2w) TC=T2w;    envf=env_t2w.sh; R="${RAW}/Dataset032_OnHarmonyT2w31"; RID=on-harmony_T2w_srcsm_20260709_122145;    W=05_13_predict_t2w_srcsm.sh ;;
    brain_dwi) TC=dwi_ap; envf=env_dwi.sh; R="${RAW}/Dataset033_OnHarmonyDWI31"; RID=on-harmony_dwi_ap_srcsm_20260921_203727; W=05_19_predict_dwi_ap_srcsm.sh ;;
  esac
  specs=(); for it in T1w T2w bold dwi_ap epi_ap gre_echo1_mag; do specs+=("${IN}/imagesTs_${it}:mr"); done
  match "brain_${TC}" "${TC}" "${R}/imagesTr" mr "${specs[@]}" || exit 1
  predict bash "${O}/05_predict/${W}" "${RID}" all
  ev "source ${O}/00_utils/${envf}; export CHECKPOINT=checkpoint_srcmatch.pth; for F in 0 1 2; do bash ${O}/06_evaluate/06_01_evaluate_testset.sh ${RID} \${F} || exit 1; done"
  audit "${M}/${TC}/auglab_${RID}" "${M}/${TC}/auglab_${RID}_srcmatch" "brain_${TC}"
  ;;
# ── Breast (ispy2 own t1wce/t2w + duke *_uniap + ispy1 t1wce/precontrast + acrin6698 dwi_uniap) ─────────────────
breast_t1wce|breast_t2w)
  BC="${T}/breast_cancer"; I="${BC}/ispy2/5_scripts_ispy2"; RAWI="${BC}/ispy2/2_nnUNet_ispy2/raw"
  if [ "${KEY}" = breast_t1wce ]; then TC=t1wce; envf=env.sh; DS=100; R="${RAWI}/Dataset100_ISPY2T1wce"; W=05_24_predict_own_t1wce_srcsm.sh; XW=05_06_predict_ispy2_t1wce_srcsm.sh
  else TC=t2w; envf=env_t2w.sh; DS=101; R="${RAWI}/Dataset101_ISPY2T2w"; W=05_27_predict_own_t2w_srcsm.sh; XW=05_12_predict_ispy2_t2w_srcsm.sh; fi
  RID="ispy2_${TC}_srcsm_20260905_163655"
  DR="${BC}/duke-breast-mri/2_nnUNet_duke-breast-mri/raw"; IR="${BC}/ispy1/2_nnUNet_ispy1/raw"; AR="${BC}/acrin6698/2_nnUNet_acrin6698/raw"
  match "breast_${TC}" "${TC}" "${R}/imagesTr" mr "${R}/imagesTs_t1wce:mr" "${R}/imagesTs_t2w:mr" \
        "${DR}/imagesTs_t1wce_uniap:mr" "${DR}/imagesTs_precontrast_uniap:mr" "${IR}/imagesTs_t1wce:mr" "${IR}/imagesTs_precontrast:mr" "${AR}/imagesTs_dwi_uniap:mr" || exit 1
  predict bash "${I}/05_predict/${W}" "${RID}" all
  predict env PREDICT_TIME_OVERRIDE=01:00:00 bash "${BC}/duke-breast-mri/5_scripts_duke-breast-mri/05_predict/${XW}" "${RID}" all t1wce_uniap precontrast_uniap
  predict bash "${BC}/ispy1/5_scripts_ispy1/05_predict/${XW}" "${RID}" all t1wce precontrast
  predict bash "${BC}/acrin6698/5_scripts_acrin6698/05_predict/${XW}" "${RID}" all dwi_uniap
  ev "source ${I}/00_utils/${envf}; bash ${I}/06_evaluate/06_01_evaluate_own_run.sh ${RID} auglab ${DS} all"
  for item in t1wce_uniap precontrast_uniap; do
    ev "export DUKE_ITEM=${item} METRICS_SUBDIR=${item}; bash ${BC}/duke-breast-mri/5_scripts_duke-breast-mri/06_evaluate/06_01_evaluate_ispy2_run.sh ${RID} auglab ${TC}"
  done
  ev "export ITEMS_OVERRIDE='t1wce precontrast'; bash ${BC}/ispy1/5_scripts_ispy1/06_evaluate/06_01_evaluate_run.sh ${RID} auglab ${TC}"
  ev "export ITEMS_OVERRIDE=dwi_uniap; bash ${BC}/acrin6698/5_scripts_acrin6698/06_evaluate/06_01_evaluate_run.sh ${RID} auglab ${TC}"
  audit "${BC}/ispy2/8_results_ispy2/02_metrics/ispy2_model/${TC}/auglab_${RID}" "${BC}/ispy2/8_results_ispy2/02_metrics/ispy2_model/${TC}/auglab_${RID}_srcmatch" "ispy2_${TC}_own"
  for x in duke-breast-mri/t1wce_uniap duke-breast-mri/precontrast_uniap ispy1/t1wce ispy1/precontrast acrin6698/dwi_uniap; do
    xm="${BC}/${x%%/*}/8_results_${x%%/*}/02_metrics/ispy2_model/${TC}/${x#*/}"
    audit "${xm}/auglab_${RID}" "${xm}/auglab_${RID}_srcmatch" "ispy2_${TC}_${x/\//_}"
  done
  ;;
# ── Pelvis (own ct + mri test items, both under Dataset130; models copied from TamIA; test cases sit in imagesTr) ──
pelvis_ct|pelvis_mri)
  P="${T}/pelvis_healthy/totalseg-pelvic/5_scripts_totalseg-pelvic"; M="${T}/pelvis_healthy/totalseg-pelvic/8_results_totalseg-pelvic/02_metrics/totalseg_pelvic_model"
  RAW="${T}/pelvis_healthy/totalseg-pelvic/2_nnUNet_totalseg-pelvic/raw"; IN="${RAW}/Dataset130_TotalsegPelvic_CT"
  if [ "${KEY}" = pelvis_ct ]; then TC=ct; TMOD=ct; R="${RAW}/Dataset130_TotalsegPelvic_CT"; RID=ct_srcsm_20260916_072434; W=05_06_predict_ct_srcsm.sh
  else TC=mri; TMOD=mr; R="${RAW}/Dataset131_TotalsegPelvic_MRI"; RID=mri_srcsm_20260916_072453; W=05_17_predict_mri_srcsm.sh; fi
  EXCLUDE_CASES="$(mkexcl "${IN}/imagesTs_${TC}")" \
    match "pelvis_${TC}" "${TC}" "${R}/imagesTr" "${TMOD}" "${IN}/imagesTs_ct:ct" "${IN}/imagesTs_mri:mr" || exit 1
  predict env TRAINING_CONTRAST="${TC}" bash "${P}/05_predict/${W}" "${RID}" all
  ev "export TRAINING_CONTRAST=${TC}; bash ${P}/06_evaluate/06_01_evaluate_run.sh ${RID} auglab"
  audit "${M}/${TC}/auglab_${RID}" "${M}/${TC}/auglab_${RID}_srcmatch" "pelvis_${TC}"
  ;;
# ── Mandible (toothfairy2 own CBCT + hanseg CT/MR-T1 + pddca CT, orientation-corrected _sif inputs; mandible-only) ─
mandible)
  MD="${T}/mandible_healthy"; TC=cbct; RID=toothfairy2_cbct_srcsm_20260908_013027
  R="${MD}/toothfairy2/2_nnUNet_toothfairy2/raw/Dataset110_ToothFairy2CBCT"; HR="${MD}/hanseg/2_nnUNet_hanseg/raw"; PR="${MD}/pddca/2_nnUNet_pddca/raw"
  EXCLUDE_CASES="$(mkexcl "${R}/imagesTs_cbct")" \
    match mandible_cbct cbct "${R}/imagesTr" ct "${R}/imagesTs_cbct:ct" "${HR}/imagesTs_ct_sif:ct" "${HR}/imagesTs_mrt1_sif:mr" "${PR}/imagesTs_ct_sif:ct" || exit 1
  predict bash "${MD}/toothfairy2/5_scripts_toothfairy2/05_predict/05_06_predict_srcsm.sh" "${RID}" all
  # hanseg/pddca: inputs <item>_sif -> matched <item>_sif_srcmatch_cbct; outputs fold{k}/sif_srcmatch/<item>
  env PREDICT_INPUT_SUFFIX=_sif_srcmatch_cbct PREDICT_OUTPUT_SUBDIR=sif_srcmatch bash "${MD}/hanseg/5_scripts_hanseg/05_predict/05_06_predict_srcsm.sh" "${RID}" all ct mrt1 \
    > "${OUTD}/${KEY}.predict_hanseg.log" 2>&1 || fail "predict hanseg"
  env PREDICT_INPUT_SUFFIX=_sif_srcmatch_cbct PREDICT_OUTPUT_SUBDIR=sif_srcmatch bash "${MD}/pddca/5_scripts_pddca/05_predict/05_06_predict_srcsm.sh" "${RID}" all ct \
    > "${OUTD}/${KEY}.predict_pddca.log" 2>&1 || fail "predict pddca"
  # these evaluators name outputs <cat>_<RID> (no tag; pddca even picks its sub-dir by pattern): point each METRICS
  # root at a temp dir so the plain SRCSM metrics can never be overwritten, then move to <cat>_<RID>_srcmatch
  TM="${MD}/toothfairy2/8_results_toothfairy2/02_metrics/toothfairy2_model/cbct"
  HM="${MD}/hanseg/8_results_hanseg/02_metrics/toothfairy2_model/cbct"; PM="${MD}/pddca/8_results_pddca/02_metrics/toothfairy2_model/cbct"
  ev "export TF2_PRED=${MD}/toothfairy2/8_results_toothfairy2/01_predictions/toothfairy2_model/cbct TF2_METRICS=${TM}/_srcmatch_tmp TF2_METRICS_SRC=${TM} \
        TF2_RAW=${R} FORCE_SUB= PRED_SUBDIR=srcmatch ONLY_RUN=${RID}; bash ${MD}/toothfairy2/5_scripts_toothfairy2/06_evaluate/06_08_evaluate_cbct.sh"
  ev "export HS_PRED=${MD}/hanseg/8_results_hanseg/01_predictions/toothfairy2_model/cbct HS_METRICS=${HM}/_srcmatch_tmp MO_DIR= FORCE_SUB= \
        GT_OVERRIDE=${HR}/labelsTs_ct_sif PRED_SUBDIR=sif_srcmatch ONLY_RUN=${RID}; bash ${MD}/hanseg/5_scripts_hanseg/06_evaluate/06_03_eval_mandible_only.sh"
  ev "export PD_PRED=${MD}/pddca/8_results_pddca/01_predictions/toothfairy2_model/cbct PD_METRICS=${PM}/_srcmatch_tmp MO_DIR= \
        GT_OVERRIDE=${PR}/labelsTs_ct_sif PRED_SUBDIR=sif_srcmatch ONLY_RUN=${RID}; bash ${MD}/pddca/5_scripts_pddca/06_evaluate/06_01_eval_mandible_only.sh"
  for X in "${TM}" "${HM}" "${PM}"; do
    T1="$(find "${X}/_srcmatch_tmp" -maxdepth 3 -type d -name "auglab_${RID}" | head -1)"
    if [ -n "${T1}" ]; then rm -rf "${X}/auglab_${RID}_srcmatch"; mv "${T1}" "${X}/auglab_${RID}_srcmatch"; rm -r "${X}/_srcmatch_tmp"
    else fail "no metrics in ${X}/_srcmatch_tmp"; fi
    audit "${X}/auglab_${RID}" "${X}/auglab_${RID}_srcmatch" "mandible_$(basename "$(dirname "$(dirname "$(dirname "${X}")")")")"
  done
  ;;
*) log "FAILED: no recipe for ${KEY}"; exit 3 ;;
esac
if [ "${FAIL}" = 0 ]; then log "DONE ${KEY}"; else log "FAILED ${KEY} (see above)"; fi
exit ${FAIL}
