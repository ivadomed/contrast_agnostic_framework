#!/usr/bin/env bash
# Run table for the FINAL ladder rung (OURS recipe + boundary PV; scripts/cluster/tamia_pack_rung7_auglab_pv.sh).
# Same row format and the same shared predict/eval job scripts as rung 6 (tamia_rung6_pv_{predict,eval}_job.sh, ROSTER=this
# file), but: category `auglab` (like every OURS run), FLAT metrics (RUNG_MSUB=""), each dataset's existing OURS predict wrapper
# (val000 mirror = checkpoint_best; its TRAINER matches the training wrapper's -- verified by `postcheck`), and RUN_IDs resolved
# from the rung-7 pack recording (no timestamps in this file). Row: name|pack|dir|env|tamia env|tc|predict wrapper|args|RUN prefix|
# expected predictions|eval kind|expected scored cases
source scripts/cluster/tamia_rung6_pv_runs.sh     # shared helpers (rung6_eval_pre_env, rung6_predict_extra_env) + dir vars/counts
P7=auglabAug_v26_6_2_pv_train050_val000
RUNG_ROWS=(
"brats_t1n|A_brats|$B|env.sh|tamia_env.sh|t1n|05_20_predict_t1n_auglabAug_v26_6_2_train050_val000.sh||brats2024-glioma_t1n_${P7}|$BR|brats:051|$BR"
"brats_t2w|A_brats|$B|env_t2w.sh|tamia_env.sh|t2w|05_22_predict_t2w_auglabAug_v26_6_2_train050_val000_dualval.sh||brats2024-glioma_t2w_${P7}|$BR|brats:052|$BR"
"brats_t2f|A_brats|$B|env_t2f.sh|tamia_env.sh|t2f|05_55_predict_t2f_auglabAug_v26_6_2_train050_val000_dualval.sh||brats2024-glioma_t2f_${P7}|$BR|brats:053|$BR"
"brats_t1c|A_brats|$B|env_t1c.sh|tamia_env.sh|t1c|05_75_predict_t1c_auglabAug_v26_6_2_train050_val000_dualval.sh||brats2024-glioma_t1c_${P7}|$BR|brats:054|$BR"
"onh_T1w|B_onharmony|$O|env.sh|tamia_env_onharmony.sh|T1w|05_08_predict_t1w_auglabAug_v26_6_2_train050_val000.sh||on-harmony_T1w_${P7}|$ONP|onh|$ONE"
"onh_T2w|B_onharmony|$O|env_t2w.sh|tamia_env_onharmony.sh|T2w|05_14_predict_t2w_auglabAug_v26_6_2_train050_val000.sh||on-harmony_T2w_${P7}|$ONP|onh|$ONE"
"onh_dwi_ap|B_onharmony|$O|env_dwi.sh|tamia_env_onharmony.sh|dwi_ap|05_20_predict_dwi_ap_auglabAug_v26_6_2_train050_val000.sh||on-harmony_dwi_ap_${P7}|$ONP|onh|$ONE"
"openms_flair|C_openms_tf2|$M|env.sh|tamia_env_openms.sh|flair|05_20_predict_auglabAug_v26_6_2_train050_val000.sh||open-ms_flair_${P7}|$OM|openms_flair|$OM"
"openms_t1w|C_openms_tf2|$M|env_t1w.sh|tamia_env_openms.sh|t1w|05_21_predict_t1w_auglabAug_v26_6_2_train050_val000.sh||open-ms_t1w_${P7}|$OM|openms_t1w|$OM"
"tf2_cbct|C_openms_tf2|$TF|env.sh|tamia_env_toothfairy2.sh|cbct|05_07_predict_ours_val000.sh||toothfairy2_cbct_${P7}|cbct:71|tf2|cbct:71"
"hanseg_sif|C_openms_tf2|$HS|env.sh|tamia_env_hanseg.sh|cbct|05_07_predict_ours_val000.sh|all ct mrt1|toothfairy2_cbct_${P7}|sif/ct:42 sif/mrt1:41|hanseg|ct:42 mrt1:41"
"pddca_sif|C_openms_tf2|$PD|env.sh|tamia_env_pddca.sh|cbct|05_07_predict_ours_val000.sh|all ct|toothfairy2_cbct_${P7}|sif/ct:40|pddca|ct:40"
"ispy2_t1wce|D_ispy2_chaos|$I|env.sh|tamia_env_ispy2.sh|t1wce|05_41_predict_own_t1wce_auglabAug_v26_6_2_train050_val000.sh||ispy2_t1wce_${P7}|$IS|ispy2:100|$IS"
"ispy2_t2w|D_ispy2_chaos|$I|env_t2w.sh|tamia_env_ispy2.sh|t2w|05_28_predict_own_t2w_auglabAug_v26_6_2_train050_val000.sh||ispy2_t2w_${P7}|$IS|ispy2:101|$IS"
"chaos_t1in|D_ispy2_chaos|$C|env.sh|tamia_env_chaos.sh|t1in|05_09_predict_auglabAug_v26_6_2_train050_val000.sh||chaos_t1in_${P7}|$CH|chaos:60|$CH"
"chaos_t2spir|D_ispy2_chaos|$C|env_t2spir.sh|tamia_env_chaos.sh|t2spir|05_34_predict_t2spir_auglabAug_v26_6_2_train050_val000.sh||chaos_t2spir_${P7}|$CH|chaos:61|$CH"
"pelvic_ct|E_pelvic|$P|env.sh|tamia_env_totalseg-pelvic.sh|ct|05_07_predict_ct_auglabAug_v26_6_2_val000.sh||totalseg-pelvic_ct_${P7}|$PV|pelvic|$PV"
"pelvic_mri|E_pelvic|$P|env_mri.sh|tamia_env_totalseg-pelvic.sh|mri|05_18_predict_mri_auglabAug_v26_6_2_val000.sh||totalseg-pelvic_mri_${P7}|$PV|pelvic|$PV"
)
RUNG_CAT=auglab; RUNG_MSUB=""
# run id from the pack recording: <PACKS_ROOT>/<pack>/fold0_<prefix>_<timestamp>.sh (unique; error if 0 or >1)
rung_resolve_run() {
    local pack="$1" prefix="$2" root; root="$(cat "${SCRATCH:-/scratch/${USER:0:1}/${USER}}/_packruns_rung7_auglab_pv_root.txt")"
    local f; f=( "${root}/${pack}/fold0_${prefix}_"[0-9]*.sh )
    [ -e "${f[0]}" ] && [ "${#f[@]}" = 1 ] || { echo "RESOLVE_FAILED(${prefix})"; return 1; }
    f="$(basename "${f[0]}" .sh)"; echo "${f#fold0_}"
}
