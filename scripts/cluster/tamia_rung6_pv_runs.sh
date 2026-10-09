#!/usr/bin/env bash
# Single source of truth for the ladder RUNG-6 (PALETTE + boundary PV) runs on TamIA:
# training RUN_IDs (from scripts/cluster/tamia_pack_rung6_pv.sh's recording, 2026-10-03) and
# how each one is predicted + evaluated -- with the SAME wrappers/settings its ladder's rung 5
# uses. Sourced by tamia_rung6_pv_predict_job.sh / tamia_rung6_pv_eval_job.sh.
#
# Row: name|pack|scripts dir (under benchmark/02_tasks)|env file|tamia env|TRAINING_CONTRAST|
#      predict wrapper (05_predict/)|extra predict args|RUN_ID|expected predictions "item:n ..."|
#      eval kind|expected scored cases "item:n ..."
# Model location for every row: ${PREDICTIONS_ROOT}/${MODEL_TYPE}/<tc>/nnUNet/<RUN_ID> (the
# predict wrappers' convention; brats t1n/t2w/t2f trained into the flat 8_results/nnUNet/ and
# are symlinked there by `tamia_pack_rung6_pv.sh queue-post`).
# Mandible external arms (hanseg, pddca) predict the S-I-FLIPPED inputs (_sif -> fold{k}/sif/),
# exactly as their rung 5 was (verified against the rung-5 outputs on TamIA scratch, 2026-10-03).
# Expected counts = the rung-5 runs' own counts (eval_all.csv on Vulcan; on-harmony gre has 8
# predictions but 7 scored cases).
B=brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma
O=brain_healthy/on-harmony/5_scripts_on-harmony
M=brain_ms/open-ms/5_scripts_open-ms
TF=mandible_healthy/toothfairy2/5_scripts_toothfairy2
HS=mandible_healthy/hanseg/5_scripts_hanseg
PD=mandible_healthy/pddca/5_scripts_pddca
I=breast_cancer/ispy2/5_scripts_ispy2
C=abdomen_healthy/chaos/5_scripts_chaos
P=pelvis_healthy/totalseg-pelvic/5_scripts_totalseg-pelvic
BR="t1n:70 t1c:70 t2w:70 t2f:70"
ONP="T1w:8 T2w:8 bold:8 dwi_ap:8 epi_ap:4 gre_echo1_mag:8"; ONE="T1w:8 T2w:8 bold:8 dwi_ap:8 epi_ap:4 gre_echo1_mag:7"
OM="flair:8 t1w:8 t2w:8"; IS="t1wce:102 t2w:168"; CH="t1in:4 t1out:4 t2spir:4 ct:20"; PV="ct:30 mri:27"
RUNG6_ROWS=(
"brats_t1n|A_brats|$B|env.sh|tamia_env.sh|t1n|05_14_predict_t1n_v26_6_2_train050_val100.sh||brats2024-glioma_t1n_v26_6_2_pv_train050_val100_20261003_111212|$BR|brats:051|$BR"
"brats_t2w|A_brats|$B|env_t2w.sh|tamia_env.sh|t2w|05_17_predict_t2w_v26_6_2_train050_val100.sh||brats2024-glioma_t2w_v26_6_2_pv_train050_val100_20261003_111212|$BR|brats:052|$BR"
"brats_t2f|A_brats|$B|env_t2f.sh|tamia_env.sh|t2f|05_57_predict_t2f_v26_6_2_train050_val100.sh||brats2024-glioma_t2f_v26_6_2_pv_train050_val100_20261003_111213|$BR|brats:053|$BR"
"brats_t1c|A_brats|$B|env_t1c.sh|tamia_env.sh|t1c|05_77_predict_t1c_v26_6_2_train050_val100.sh||brats2024-glioma_t1c_v26_6_2_pv_train050_val100_20261003_111213|$BR|brats:054|$BR"
"onh_T1w|B_onharmony|$O|env.sh|tamia_env_onharmony.sh|T1w|05_21_predict_t1w_v26_6_2_train050_val100.sh||on-harmony_T1w_v26_6_2_pv_train050_val100_20261003_111214|$ONP|onh|$ONE"
"onh_T2w|B_onharmony|$O|env_t2w.sh|tamia_env_onharmony.sh|T2w|05_22_predict_t2w_v26_6_2_train050_val100.sh||on-harmony_T2w_v26_6_2_pv_train050_val100_20261003_111214|$ONP|onh|$ONE"
"onh_dwi_ap|B_onharmony|$O|env_dwi.sh|tamia_env_onharmony.sh|dwi_ap|05_23_predict_dwi_ap_v26_6_2_train050_val100.sh||on-harmony_dwi_ap_v26_6_2_pv_train050_val100_20261003_111214|$ONP|onh|$ONE"
"openms_flair|C_openms_tf2|$M|env.sh|tamia_env_openms.sh|flair|05_24_predict_v26_6_2_train050_val000.sh||open-ms_flair_v26_6_2_pv_train050_val000_20261003_111215|$OM|openms_flair|$OM"
"openms_t1w|C_openms_tf2|$M|env_t1w.sh|tamia_env_openms.sh|t1w|05_15_predict_t1w_v26_6_2_train050_val100.sh||open-ms_t1w_v26_6_2_pv_train050_val100_20261003_111215|$OM|openms_t1w|$OM"
"tf2_cbct|C_openms_tf2|$TF|env.sh|tamia_env_toothfairy2.sh|cbct|05_12_predict_v26_6_2_train050_val100.sh||toothfairy2_cbct_v26_6_2_pv_train050_val100_20261003_111216|cbct:71|tf2|cbct:71"
"hanseg_sif|C_openms_tf2|$HS|env.sh|tamia_env_hanseg.sh|cbct|05_12_predict_v26_6_2_train050_val100.sh|all ct mrt1|toothfairy2_cbct_v26_6_2_pv_train050_val100_20261003_111216|sif/ct:42 sif/mrt1:41|hanseg|ct:42 mrt1:41"
"pddca_sif|C_openms_tf2|$PD|env.sh|tamia_env_pddca.sh|cbct|05_12_predict_v26_6_2_train050_val100.sh|all ct|toothfairy2_cbct_v26_6_2_pv_train050_val100_20261003_111216|sif/ct:40|pddca|ct:40"
"ispy2_t1wce|D_ispy2_chaos|$I|env.sh|tamia_env_ispy2.sh|t1wce|05_42_predict_own_t1wce_v26_6_2_train050_val100_ladder.sh||ispy2_t1wce_v26_6_2_pv_train050_val100_20261003_111216|$IS|ispy2:100|$IS"
"ispy2_t2w|D_ispy2_chaos|$I|env_t2w.sh|tamia_env_ispy2.sh|t2w|05_48_predict_own_t2w_v26_6_2_train050_val100_ladder.sh||ispy2_t2w_v26_6_2_pv_train050_val100_20261003_111217|$IS|ispy2:101|$IS"
"chaos_t1in|D_ispy2_chaos|$C|env.sh|tamia_env_chaos.sh|t1in|05_38_predict_t1in_v26_6_2.sh||chaos_t1in_v26_6_2_pv_train050_val000_20261003_111217|$CH|chaos:60|$CH"
"chaos_t2spir|D_ispy2_chaos|$C|env_t2spir.sh|tamia_env_chaos.sh|t2spir|05_22_predict_t2spir_v26_6_2_train050_val100.sh||chaos_t2spir_v26_6_2_pv_train050_val100_20261003_111218|$CH|chaos:61|$CH"
"pelvic_ct|E_pelvic|$P|env.sh|tamia_env_totalseg-pelvic.sh|ct|05_12_predict_ct_ladder_v26_6_2_train050_val100.sh||totalseg-pelvic_ct_v26_6_2_pv_train050_val100_20261003_120315|$PV|pelvic|$PV"
"pelvic_mri|E_pelvic|$P|env_mri.sh|tamia_env_totalseg-pelvic.sh|mri|05_23_predict_mri_ladder_v26_6_2_train050_val100.sh||totalseg-pelvic_mri_v26_6_2_pv_train050_val100_20261003_120315|$PV|pelvic|$PV"
)
# Generic names read by the shared predict/eval job scripts (tamia_rung6_pv_{predict,eval}_job.sh). Another roster
# (e.g. tamia_rung7_runs.sh) overrides these after sourcing this file. RUNG_CAT = nnUNet-results category dir,
# RUNG_MSUB = metrics sub-dir ("ablations" for ladder-exclusive runs, "" = flat headline layout),
# rung_resolve_run <pack> <run field> = turn a run-id PREFIX into the real run id (identity here: ids are literal).
RUNG_ROWS=("${RUNG6_ROWS[@]}"); RUNG_CAT=nnUNet; RUNG_MSUB=ablations
rung_resolve_run() { echo "$2"; }
# Exported BEFORE a row's env files are sourced, for eval (common_env.sh keeps an already-exported
# METRICS_ROOT; brats' tamia_env.sh does not override it and the default is the file-count-limited /project).
rung6_eval_pre_env() {
    case "$1" in
        brats:*) export METRICS_ROOT="${SCRATCH:-/scratch/${USER:0:1}/${USER}}/brats2024-glioma/8_results/02_metrics" ;;
    esac
}
# Prediction env extras per row (flipped mandible arms only).
rung6_predict_extra_env() {
    case "$1" in
        hanseg_sif|pddca_sif) export PREDICT_INPUT_SUFFIX=_sif PREDICT_OUTPUT_SUBDIR=sif ;;
    esac
}
