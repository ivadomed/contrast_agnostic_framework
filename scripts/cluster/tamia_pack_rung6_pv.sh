#!/usr/bin/env bash
# TamIA launcher: causal-ladder RUNG 6 (PALETTE + boundary partial-volume) for all 14
# ladder training sets x 3 folds = 42 fold trainings, packed onto 4 whole-node 4xH100 jobs.
# Each fold is recorded by its own dataset's rung-6 wrapper (04_XX_train_<c>_v26_6_2_pv_*.sh,
# a copy of that ladder's rung-5 wrapper + the pv train config), so all train settings
# (trainer, epochs, val config) come from the wrapper, not from here.
#
#   cd /project/aip-jcohen/paulh/mri_synthesis_project
#   bash scripts/cluster/tamia_pack_rung6_pv.sh record   # record all packs + verify cmd files (no submit)
#   bash scripts/cluster/tamia_pack_rung6_pv.sh submit   # submit the recorded packs (chain-resumed)
#
# Pack layout (densities follow this project's earlier ladder packs on the same datasets):
#   A brats (t1n,t2w,t2f,t1c; 2500 ep)        12 folds  chain 8
#   B on-harmony (T1w,T2w,dwi_ap; 2000 ep)      9 folds  chain 4
#   C open-ms (flair,t1w; 2000 ep) + toothfairy2 (cbct; 1000 ep)   9 folds  chain 4
#   D ispy2 (t1wce,t2w; 1000 ep) + chaos (t1in,t2spir; 200 ep)    12 folds  chain 4
# Pack dirs are fixed per launch (PACKS_ROOT below, persisted in PACKS_ROOT.txt) so
# `submit`/re-`submit` (to extend a chain) reuse the same recording.
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
export SCRATCH="${SCRATCH:-/scratch/p/paulh}"
T=benchmark/02_tasks
STATE="${SCRATCH}/_packruns_rung6_pv_root.txt"
MODE="${1:?usage: record|submit}"

if [ "${MODE}" = "record" ]; then
    [ -f "${STATE}" ] && { echo "already recorded: $(cat "${STATE}") (delete ${STATE} to re-record)"; exit 1; }
    PACKS_ROOT="${SCRATCH}/_packruns/rung6_pv_$(date +%Y%m%d_%H%M%S)"
    echo "${PACKS_ROOT}" > "${STATE}"
else
    PACKS_ROOT="$(cat "${STATE}")"
fi
echo "[rung6] PACKS_ROOT=${PACKS_ROOT}"

# record <pack> <wrapper path> <tamia env file>
record() {
    local pack="$1" wrapper="$2" tenv="$3"
    local envf; envf="$(grep -o 'source "$(dirname "$0")/../00_utils/env[a-z0-9_]*\.sh"' "${wrapper}" | sed 's|.*/00_utils/||; s|"$||')"
    echo "[record] ${pack} <- $(basename "${wrapper}")  (env ${envf}, ${tenv})"
    (
        # shellcheck disable=SC1090
        source "$(dirname "${wrapper}")/../00_utils/${envf}"
        # shellcheck disable=SC1090
        source "scripts/cluster/${tenv}"
        # The wrapper re-sources its env*.sh, which re-exports nnUNet_results to the
        # /project tree (TamIA /project is file-count-limited). train_common.sh honours
        # NNUNET_RESULTS_BASE first, so pin the tamia (scratch) value here.
        export NNUNET_RESULTS_BASE="${nnUNet_results}"
        RUN_JOB_PACK_DIR="${PACKS_ROOT}/${pack}" bash "${wrapper}"
    )
}

if [ "${MODE}" = "record" ]; then
    B=$T/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/04_train
    record A_brats "$B/04_81_train_t1n_v26_6_2_pv_train050_val100.sh" tamia_env.sh
    record A_brats "$B/04_82_train_t2w_v26_6_2_pv_train050_val100.sh" tamia_env.sh
    record A_brats "$B/04_83_train_t2f_v26_6_2_pv_train050_val100.sh" tamia_env.sh
    record A_brats "$B/04_84_train_t1c_v26_6_2_pv_train050_val100.sh" tamia_env.sh
    O=$T/brain_healthy/on-harmony/5_scripts_on-harmony/04_train
    record B_onharmony "$O/04_47_train_t1w_v26_6_2_pv_train050_val100.sh"    tamia_env_onharmony.sh
    record B_onharmony "$O/04_48_train_t2w_v26_6_2_pv_train050_val100.sh"    tamia_env_onharmony.sh
    record B_onharmony "$O/04_49_train_dwi_ap_v26_6_2_pv_train050_val100.sh" tamia_env_onharmony.sh
    M=$T/brain_ms/open-ms/5_scripts_open-ms/04_train
    record C_openms_tf2 "$M/04_39_train_flair_v26_6_2_pv_train050_val000.sh" tamia_env_openms.sh
    record C_openms_tf2 "$M/04_40_train_t1w_v26_6_2_pv_train050_val100.sh"   tamia_env_openms.sh
    record C_openms_tf2 "$T/mandible_healthy/toothfairy2/5_scripts_toothfairy2/04_train/04_15_train_cbct_v26_6_2_pv_train050_val100.sh" tamia_env_toothfairy2.sh
    I=$T/breast_cancer/ispy2/5_scripts_ispy2/04_train
    record D_ispy2_chaos "$I/04_26_train_t1wce_v26_6_2_pv_train050_val100.sh" tamia_env_ispy2.sh
    record D_ispy2_chaos "$I/04_27_train_t2w_v26_6_2_pv_train050_val100.sh"   tamia_env_ispy2.sh
    C=$T/abdomen_healthy/chaos/5_scripts_chaos/04_train
    record D_ispy2_chaos "$C/04_59_train_t1in_v26_6_2_pv_train050_val000.sh"   tamia_env_chaos.sh
    record D_ispy2_chaos "$C/04_60_train_t2spir_v26_6_2_pv_train050_val100.sh" tamia_env_chaos.sh

    echo; echo "=== verification (every recorded fold) ==="
    bad=0
    declare -A EXPECT=([A_brats]=12 [B_onharmony]=9 [C_openms_tf2]=9 [D_ispy2_chaos]=12)
    for p in A_brats B_onharmony C_openms_tf2 D_ispy2_chaos; do
        n=$(wc -l < "${PACKS_ROOT}/$p/index.tsv"); echo "-- $p: ${n} folds (expect ${EXPECT[$p]})"
        [ "$n" = "${EXPECT[$p]}" ] || { echo "   FOLD COUNT MISMATCH"; bad=1; }
        while IFS=$'\t' read -r cmd log name done; do
            # cmd files hold a printf-%q'd  bash -c $'...\n export X=\'v\' ...'  string:
            # turn literal \n into newlines and drop the \' quoting before parsing.
            clean="$(sed -e 's/\\n/\n/g' -e "s/\\\\'//g" "$cmd")"
            raw=$(printf '%s\n' "$clean" | sed -n 's/^ *export nnUNet_preprocessed=//p' | head -1)
            res=$(printf '%s\n' "$clean" | sed -n 's/^ *export nnUNet_results=//p' | head -1)
            cfg=$(printf '%s\n' "$clean" | sed -n 's/^ *export AUGLAB_PARAMS_GPU_JSON=//p' | head -1 | sed 's|.*/||')
            ok=OK
            case "$raw" in /scratch/*) [ -d "$raw" ] || ok="NO_PREPROC_DIR";; *) ok="PREPROC_NOT_SCRATCH";; esac
            case "$res" in /scratch/*) ;; *) ok="${ok},RESULTS_NOT_SCRATCH";; esac
            [ "$cfg" = "transform_params_gpu_v26_6_2_pv_synth_spatialDA_train050.json" ] || ok="${ok},WRONG_CFG($cfg)"
            grep -q "nnUNetv2_train" "$cmd" || ok="${ok},NO_TRAIN_CMD"
            grep -q "v26_6_2_pv_" <<<"$res" || ok="${ok},RESULTS_NOT_PV_RUN"
            [ "$ok" = OK ] || bad=1
            printf '   %-6s %s\n      preproc=%s\n      results=%s\n' "$ok" "$name" "$raw" "$res"
        done < "${PACKS_ROOT}/$p/index.tsv"
    done
    [ "$bad" = 0 ] && echo "VERIFY PASSED" || echo "VERIFY FAILED"
    exit 0
fi

# submit
submit() {
    local pack="$1" chain="$2"
    PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_TIME="${PACK_TIME:-23:59:00}" PACK_CHAIN="${chain}" \
    PACK_USE_MPS=0 PACK_JOB_NAME="rung6pv_${pack}" \
      bash scripts/job_runner/run_job_pack_submit.sh "${PACKS_ROOT}/${pack}"
    sleep 5
}
submit A_brats 8
submit B_onharmony 4
submit C_openms_tf2 4
submit D_ispy2_chaos 4
