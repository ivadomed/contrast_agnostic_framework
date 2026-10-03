#!/usr/bin/env bash
# TamIA launcher: causal-ladder RUNG 6 (PALETTE + boundary partial-volume) for all 16
# ladder training sets x 3 folds = 48 fold trainings, packed onto 5 whole-node 4xH100 jobs.
# Each fold is recorded by its own dataset's rung-6 wrapper (04_XX_train_<c>_v26_6_2_pv_*.sh,
# a copy of that ladder's rung-5 wrapper + the pv train config), so all train settings
# (trainer, epochs, val config) come from the wrapper, not from here.
#
#   cd /project/aip-jcohen/paulh/mri_synthesis_project
#   bash scripts/cluster/tamia_pack_rung6_pv.sh record [PACK...]  # record (default: all) + verify, no submit
#   bash scripts/cluster/tamia_pack_rung6_pv.sh submit [PACK...]  # submit those packs (chain-resumed)
# A pack whose index.tsv already exists is never re-recorded.
#
# Pack layout (densities follow this project's earlier ladder packs on the same datasets):
#   A brats (t1n,t2w,t2f,t1c; 2500 ep)        12 folds  chain 8
#   B on-harmony (T1w,T2w,dwi_ap; 2000 ep)      9 folds  chain 4
#   C open-ms (flair,t1w; 2000 ep) + toothfairy2 (cbct; 1000 ep)   9 folds  chain 4
#   D ispy2 (t1wce,t2w; 1000 ep) + chaos (t1in,t2spir; 200 ep)    12 folds  chain 4
#   E totalseg-pelvic (ct,mri; 200 ep)                              6 folds  chain 2
#     (no 06_1X ladder script yet, but all its rungs are trained -- added 2026-10-03)
# Pack dirs are fixed per launch (PACKS_ROOT below, persisted in PACKS_ROOT.txt) so
# `submit`/re-`submit` (to extend a chain) reuse the same recording.
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
export SCRATCH="${SCRATCH:-/scratch/p/paulh}"
T=benchmark/02_tasks
STATE="${SCRATCH}/_packruns_rung6_pv_root.txt"
MODE="${1:?usage: record|submit [PACK...]}"; shift
ALL_PACKS="A_brats B_onharmony C_openms_tf2 D_ispy2_chaos E_pelvic"
SEL="${*:-${ALL_PACKS}}"
want() { [[ " ${SEL} " == *" $1 "* ]] && { [ "${MODE}" != record ] || [ ! -f "${PACKS_ROOT}/$1/index.tsv" ]; }; }

if [ -f "${STATE}" ]; then
    PACKS_ROOT="$(cat "${STATE}")"
else
    [ "${MODE}" = record ] || { echo "nothing recorded yet"; exit 1; }
    PACKS_ROOT="${SCRATCH}/_packruns/rung6_pv_$(date +%Y%m%d_%H%M%S)"
    echo "${PACKS_ROOT}" > "${STATE}"
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
    if want A_brats; then
    record A_brats "$B/04_81_train_t1n_v26_6_2_pv_train050_val100.sh" tamia_env.sh
    record A_brats "$B/04_82_train_t2w_v26_6_2_pv_train050_val100.sh" tamia_env.sh
    record A_brats "$B/04_83_train_t2f_v26_6_2_pv_train050_val100.sh" tamia_env.sh
    record A_brats "$B/04_84_train_t1c_v26_6_2_pv_train050_val100.sh" tamia_env.sh
    fi
    O=$T/brain_healthy/on-harmony/5_scripts_on-harmony/04_train
    if want B_onharmony; then
    record B_onharmony "$O/04_47_train_t1w_v26_6_2_pv_train050_val100.sh"    tamia_env_onharmony.sh
    record B_onharmony "$O/04_48_train_t2w_v26_6_2_pv_train050_val100.sh"    tamia_env_onharmony.sh
    record B_onharmony "$O/04_49_train_dwi_ap_v26_6_2_pv_train050_val100.sh" tamia_env_onharmony.sh
    fi
    M=$T/brain_ms/open-ms/5_scripts_open-ms/04_train
    if want C_openms_tf2; then
    record C_openms_tf2 "$M/04_39_train_flair_v26_6_2_pv_train050_val000.sh" tamia_env_openms.sh
    record C_openms_tf2 "$M/04_40_train_t1w_v26_6_2_pv_train050_val100.sh"   tamia_env_openms.sh
    record C_openms_tf2 "$T/mandible_healthy/toothfairy2/5_scripts_toothfairy2/04_train/04_15_train_cbct_v26_6_2_pv_train050_val100.sh" tamia_env_toothfairy2.sh
    fi
    I=$T/breast_cancer/ispy2/5_scripts_ispy2/04_train
    if want D_ispy2_chaos; then
    record D_ispy2_chaos "$I/04_26_train_t1wce_v26_6_2_pv_train050_val100.sh" tamia_env_ispy2.sh
    record D_ispy2_chaos "$I/04_27_train_t2w_v26_6_2_pv_train050_val100.sh"   tamia_env_ispy2.sh
    C=$T/abdomen_healthy/chaos/5_scripts_chaos/04_train
    record D_ispy2_chaos "$C/04_59_train_t1in_v26_6_2_pv_train050_val000.sh"   tamia_env_chaos.sh
    record D_ispy2_chaos "$C/04_60_train_t2spir_v26_6_2_pv_train050_val100.sh" tamia_env_chaos.sh
    fi
    P=$T/pelvis_healthy/totalseg-pelvic/5_scripts_totalseg-pelvic/04_train
    if want E_pelvic; then
    record E_pelvic "$P/04_23_train_ct_v26_6_2_pv_train050_val100.sh"  tamia_env_totalseg-pelvic.sh
    record E_pelvic "$P/04_24_train_mri_v26_6_2_pv_train050_val100.sh" tamia_env_totalseg-pelvic.sh
    fi

    echo; echo "=== verification (every recorded fold) ==="
    bad=0
    declare -A EXPECT=([A_brats]=12 [B_onharmony]=9 [C_openms_tf2]=9 [D_ispy2_chaos]=12 [E_pelvic]=6)
    for p in ${SEL}; do
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

# ---- post-training: predict + evaluate (run table: scripts/cluster/tamia_rung6_pv_runs.sh) ----
#   bash scripts/cluster/tamia_pack_rung6_pv.sh postcheck            # resolve every row's env, no submit
#   bash scripts/cluster/tamia_pack_rung6_pv.sh queue-post [PACK...] # predict job per pack + eval job per row
if [ "${MODE}" = postcheck ] || [ "${MODE}" = queue-post ]; then
    source scripts/cluster/tamia_rung6_pv_runs.sh
    bad=0
    for r in "${RUNG6_ROWS[@]}"; do IFS='|' read -r name pack dir envf tenv tc wrapper args run pexp ekind eexp <<<"$r"
        [[ " ${SEL} " == *" ${pack} "* ]] || continue
        line="$( rung6_eval_pre_env "$ekind"; export TRAINING_CONTRAST="$tc"
                 source "$T/$dir/00_utils/$envf" >/dev/null 2>&1; source "scripts/cluster/$tenv" >/dev/null 2>&1
                 echo "${PREDICTIONS_ROOT}|${METRICS_ROOT}|${MODEL_TYPE:-${TF2_MODEL_TYPE:-}}" )"
        IFS='|' read -r pr mr mt <<<"$line"
        ok=OK
        [ -f "$T/$dir/05_predict/${wrapper}" ] || ok="${ok},NO_PREDICT_WRAPPER"
        case "$pr" in /scratch/*) ;; *) ok="${ok},PRED_NOT_SCRATCH";; esac
        case "$ekind" in tf2|hanseg|pddca) ;; *) case "$mr" in /scratch/*) ;; *) ok="${ok},METRICS_NOT_SCRATCH";; esac;; esac
        model="${pr}/${mt}/${tc}/nnUNet/${run}"
        flat="${SCRATCH}/brats2024-glioma/8_results/nnUNet/${run}"   # brats t1n/t2w/t2f: linked in at queue-post
        case "$name" in *_sif) ;; *) [ -e "${model}" ] || { [ "$pack" = A_brats ] && [ -d "${flat}" ]; } || ok="${ok},MODEL_DIR_ABSENT(${model})";; esac
        [ "$ok" = OK ] || bad=1
        printf '%-6s %-13s %-14s pred=%s\n       metrics=%s model=%s\n' "${ok%%,*}" "$name" "$ekind" "$pr" "$mr" "$model"
        [ "$ok" = OK ] || echo "       PROBLEMS: ${ok}"
    done
    [ "${MODE}" = postcheck ] && { [ "$bad" = 0 ] && echo "POSTCHECK PASSED" || echo "POSTCHECK FAILED"; exit 0; }
    [ "$bad" = 0 ] || { echo "postcheck failed -- not queueing"; exit 1; }

    # brats t1n/t2w/t2f trained into the flat 8_results/nnUNet/ (tamia_env.sh) -- link them where the
    # predict wrappers look (t1c's wrapper already pinned the co-located path).
    BS="${SCRATCH}/brats2024-glioma/8_results"
    for r in "${RUNG6_ROWS[@]}"; do IFS='|' read -r name pack dir envf tenv tc wrapper args run _ <<<"$r"
        [ "$pack" = A_brats ] && [[ " ${SEL} " == *" A_brats "* ]] || continue
        dst="${BS}/01_predictions/brats2024_glioma_model/${tc}/nnUNet/${run}"
        if [ -e "${dst}" ]; then echo "[queue-post] exists: ${dst}"
        elif [ -d "${BS}/nnUNet/${run}" ]; then mkdir -p "$(dirname "${dst}")"; ln -s "${BS}/nnUNet/${run}" "${dst}"; echo "[queue-post] linked ${dst}"
        else echo "[queue-post] ERROR: brats run dir not found: ${BS}/nnUNet/${run}"; exit 1; fi
    done

    cpu_submit() { local out; if out="$(sbatch --parsable "$@" 2>/dev/null)"; then echo "${out}"; return 0; fi
                   echo "  (bare sbatch failed -- retrying with --partition=cpubase_bynode_b1)" >&2
                   sbatch --parsable --partition=cpubase_bynode_b1 "$@"; }
    for p in ${SEL}; do
        last="$(squeue -u "${USER}" -h -n "rung6pv_${p}" -o %i | sort -n | tail -1)"
        dep=(); [ -n "${last}" ] && dep=(--dependency=afterany:"${last}")
        OUT="${PACKS_ROOT}/post_${p}"; mkdir -p "${OUT}"
        PJ="$(sbatch --parsable "${dep[@]}" --job-name="rung6pv_predict_${p}" --time=06:00:00 \
              --output="${OUT}/predict_job_%j.out" --export="ALL,PACK=${p},OUT=${OUT}" scripts/cluster/tamia_rung6_pv_predict_job.sh)"
        echo "${PJ}" > "${OUT}/PREDICT_JOB"; echo "[queue-post] ${p}: predict job ${PJ} (after training job ${last:-none, already finished})"
        for r in "${RUNG6_ROWS[@]}"; do IFS='|' read -r name pack _ <<<"$r"; [ "$pack" = "$p" ] || continue
            EJ="$(cpu_submit --dependency=afterany:"${PJ}" --job-name="rung6pv_eval_${name}" --time=05:00:00 --cpus-per-task=16 \
                  --mem=96G --account=aip-jcohen --output="${OUT}/eval_${name}_%j.out" --export="ALL,ROW=${name},OUT=${OUT}" \
                  scripts/cluster/tamia_rung6_pv_eval_job.sh)"
            echo "${EJ}" >> "${OUT}/EVAL_JOBS"; echo "[queue-post]   eval ${name}: job ${EJ}"
            sleep 2
        done
    done
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
declare -A CHAIN=([A_brats]=8 [B_onharmony]=4 [C_openms_tf2]=4 [D_ispy2_chaos]=4 [E_pelvic]=2)
for p in ${SEL}; do submit "$p" "${CHAIN[$p]}"; done
