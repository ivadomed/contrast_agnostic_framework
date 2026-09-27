#!/usr/bin/env bash
# TAMIA launcher for the FULL on-harmony dwi_ap suite (3rd training modality, 2026-09-21):
# the 6 headline methods + the causal-ablation ladder rungs, 3 folds each (0 1 2) = 30
# folds, 2000 epochs, PLUS predict+eval chained behind training.
#
#   headline : baseline(04_36) synthseg_noEM(04_38) synthseg_EM(04_37) auglab_default(04_39)
#              srcsm(04_40)  OURS DualVal(04_41 -> val000 + val100 mirrors)
#   ladder   : v26_6_2 alone/real-fill(04_42)  kmeans(04_43)  +label_remap(04_44)  +voronoi(04_45)
#
# LAYOUT (3 whole 4xH100 nodes, one dependency CHAIN each; MPS OFF — MPS hung on-harmony packs).
# Placement is explicit (PACK_GPU_MAP) and load-balanced from MEASURED per-method epoch
# times of prior on-harmony packs on this cluster (s/epoch: baseline 27, auglab_default 61,
# ladder rungs/OURS/v26 ~57, synthseg_EM 104, srcsm 174 = ~3x). synthseg_noEM was NOT
# measured; assumed = synthseg_EM (same trainer/aug family). srcsm folds each get a GPU to
# themselves so a srcsm fold never becomes the straggler that holds a node open.
# The dwi patch [112,80,112] batch 3 is ~20% fewer voxels/iter than T2w's [128,128,112]
# batch 2, so those numbers are an upper bound. Verify per-fold epoch time from
# fold_N/training_log_*.txt in the first hour and adjust chain length if it is way off.
#
# EVAL: a 4th whole-node pack of (RUN_ID, fold) worker units calling
# 06_evaluate/06_01_evaluate_testset.sh <RUN_ID> <FOLD> directly (bare 06_01 launcher fans out
# --gpus 1 jobs, which TamIA rejects). Chained with PACK_DEPENDENCY=afterany:<last job of
# each training chain>. checkpoint_final (matches every existing on-harmony headline table;
# best-ckpt is a separate backfill arm). OURS is evaluated as BOTH the _val000_ and the
# hard-linked _val100_ sibling RUN_ID (identical final ckpt, but the 7-row table needs both).
#
# REQUIRES: run ON tamia. Do NOT pre-source env files — this script sources
# env_dwi.sh THEN tamia_env_onharmony.sh itself (order matters: env.sh re-exports
# nnUNet_results unconditionally).
#
# Usage:
#   DRY_RUN=1 bash 04_46_tamia_pack_dwi_ap_suite.sh     # record + verify, submit NOTHING
#   bash 04_46_tamia_pack_dwi_ap_suite.sh               # launch
#   BASE=<existing> bash 04_46_...                      # re-use RUN_IDs / recording, extend chains
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"
S5="${ROOT}/datasets/on-harmony/5_scripts_on-harmony"

export SCRATCH="${SCRATCH:-/scratch/p/paulh}"
source "${S5}/00_utils/env_dwi.sh"
source "${ROOT}/scripts/cluster/tamia_env_onharmony.sh"
cd "${ROOT}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
BASE="${BASE:-${SCRATCH}/on-harmony/_packruns/dwi_ap_suite_${TS}}"   # dwi_ap-qualified: never share a PACK_DIR across contrasts
mkdir -p "${BASE}"
DRY_RUN="${DRY_RUN:-0}"
PACK_CHAIN="${PACK_CHAIN:-5}"
echo "[dwi-suite] BASE=${BASE}  chain=${PACK_CHAIN}x23:59  DRY_RUN=${DRY_RUN}"

# key : wrapper : METHOD (the METHOD string in the wrapper — it forms the RUN_ID) : results category
declare -A WRAP METH CAT
add() { WRAP[$1]="$2"; METH[$1]="$3"; CAT[$1]="$4"; }
add baseline       04_36_train_dwi_ap_baseline.sh                                baseline                                  nnUNet
add synthseg_EM    04_37_train_dwi_ap_synthseg_EM.sh                             synthseg_EM                               auglab
add synthseg_noEM  04_38_train_dwi_ap_synthseg_noEM.sh                           synthseg_noEM                             auglab
add auglab_default 04_39_train_dwi_ap_auglab_default.sh                          auglab_default                            auglab
add srcsm          04_40_train_dwi_ap_srcsm.sh                                   srcsm                                     auglab
add ours           04_41_train_dwi_ap_auglabAug_v26_6_2_dualval.sh               auglabAug_v26_6_2_train050_val000         auglab
add v26            04_42_train_dwi_ap_v26_6_2_train050_val100.sh                 v26_6_2_train050_val100                   nnUNet
add kmeans         04_43_train_dwi_ap_baseline_kmeans.sh                         baseline_kmeans                           auglab
add remap          04_44_train_dwi_ap_baseline_kmeans_label_remap.sh             baseline_kmeans_label_remap               auglab
add voronoi        04_45_train_dwi_ap_baseline_kmeans_label_remap_voronoi.sh     baseline_kmeans_label_remap_voronoi       auglab
KEYS="baseline synthseg_EM synthseg_noEM auglab_default srcsm ours v26 kmeans remap voronoi"

# Fixed RUN_IDs, persisted once so every job in every chain resumes the SAME run dirs.
RUNIDS="${BASE}/RUN_IDS.env"
if [ -f "${RUNIDS}" ]; then source "${RUNIDS}"; echo "[dwi-suite] reusing ${RUNIDS}"
else
    for k in ${KEYS}; do echo "RID_${k}=on-harmony_dwi_ap_${METH[$k]}_${TS}"; done > "${RUNIDS}"
    source "${RUNIDS}"
fi
sed 's/^/  /' "${RUNIDS}"

# Layout: pack -> ordered "key:fold" rows; row i is pinned to GPU (i-th entry of GPUMAP[pack]).
declare -A ROWS GPUMAP
ROWS[0]="srcsm:0 srcsm:1 srcsm:2 synthseg_noEM:0 v26:0 baseline:0";                    GPUMAP[0]="0 1 2 3 3 3"
ROWS[1]="synthseg_noEM:1 v26:1 baseline:1 synthseg_noEM:2 v26:2 baseline:2 synthseg_EM:0 kmeans:0 voronoi:0 synthseg_EM:1 kmeans:1 voronoi:1"
GPUMAP[1]="0 0 0 1 1 1 2 2 2 3 3 3"
ROWS[2]="synthseg_EM:2 kmeans:2 voronoi:2 auglab_default:0 ours:0 remap:0 auglab_default:1 ours:1 remap:1 auglab_default:2 ours:2 remap:2"
GPUMAP[2]="0 0 0 1 1 1 2 2 2 3 3 3"

LAST_IDS=""
for P in 0 1 2; do
    PD="${BASE}/pack${P}"; mkdir -p "${PD}"
    if [ -f "${PD}/index.tsv" ]; then echo "[dwi-suite] pack${P}: index.tsv exists — reusing recording"
    else
        for row in ${ROWS[$P]}; do
            k="${row%%:*}"; f="${row##*:}"; rid_var="RID_${k}"
            SINGLE_FOLD="${f}" RUN_JOB_PACK_DIR="${PD}" bash "${HERE}/${WRAP[$k]}" "${!rid_var}" > "${PD}/_rec_${k}_f${f}.out" 2>&1 \
                || { echo "RECORD FAILED: ${k} fold ${f} (see ${PD}/_rec_${k}_f${f}.out)"; exit 1; }
        done
    fi
    n_expect="$(wc -w <<< "${ROWS[$P]}")"; n_have="$(grep -c . "${PD}/index.tsv")"
    [ "${n_have}" = "${n_expect}" ] || { echo "ERROR pack${P}: recorded ${n_have} != expected ${n_expect}"; exit 1; }
    echo "[dwi-suite] pack${P}: ${n_have} folds recorded"
    # every done-marker must be on scratch (never /project — 498K/500K files)
    if cut -f4 "${PD}/index.tsv" | grep -vq "^${SCRATCH}/"; then echo "ERROR pack${P}: a donefile is not under ${SCRATCH}"; exit 1; fi
done

# ── EVAL pack: one worker unit per (RUN_ID, fold); OURS also as its _val100_ sibling ──────────
EV="${BASE}/evalpack"; mkdir -p "${EV}"
if [ ! -f "${EV}/index.tsv" ]; then
    : > "${EV}/index.tsv"
    ev_unit() {   # $1=RUN_ID  $2=category  $3=fold
        local rid="$1" cat="$2" f="$3" name="onheval_${1}_f${3}"   # FULL RUN_ID: truncating collides remap/voronoi and val000/val100
        local cmd="${EV}/${name}.sh"
        {   echo "#!/bin/bash"
            echo "export WANDB_MODE=offline nnUNet_wandb_mode=offline SCRATCH='${SCRATCH}' TRAINING_CONTRAST=dwi_ap"
            echo "source '${S5}/00_utils/env_dwi.sh'; source '${ROOT}/scripts/cluster/tamia_env_onharmony.sh'"
            echo "bash '${S5}/06_evaluate/06_01_evaluate_testset.sh' '${rid}' '${f}'"
        } > "${cmd}"
        printf '%s\t%s\t%s\t%s\n' "${cmd}" "${EV}/${name}.log" "${name}" \
            "${METRICS_ROOT}/on_harmony_model/dwi_ap/${cat}_${rid}/fold${f}/eval_all.csv" >> "${EV}/index.tsv"
    }
    for k in ${KEYS}; do
        rid_var="RID_${k}"; rid="${!rid_var}"
        for f in 0 1 2; do
            ev_unit "${rid}" "${CAT[$k]}" "${f}"
            [ "${k}" = ours ] && ev_unit "${rid/_val000_/_val100_}" auglab "${f}"
        done
    done
fi
echo "[dwi-suite] evalpack: $(grep -c . "${EV}/index.tsv") worker units (expect 33)"

if [ "${DRY_RUN}" = "1" ]; then
    echo "[dwi-suite] DRY_RUN=1 — recorded + verified, NOTHING submitted. BASE=${BASE}"; exit 0
fi

# ── submit the 3 training chains, then the eval pack behind all of them ──────────────────────
for P in 0 1 2; do
    out="$(PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_TIME=23:59:00 PACK_CHAIN="${PACK_CHAIN}" PACK_USE_MPS=0 \
           PACK_GPU_MAP="${GPUMAP[$P]}" PACK_JOB_NAME="onh_dwi_p${P}" \
           bash "${ROOT}/scripts/job_runner/run_job_pack_submit.sh" "${BASE}/pack${P}")"
    echo "${out}"
    LAST_IDS="${LAST_IDS}:$(grep -oE 'last job id=[0-9]+' <<< "${out}" | grep -oE '[0-9]+')"
done
echo "[dwi-suite] training chains' last job ids:${LAST_IDS}"
PACK_DEPENDENCY="afterany${LAST_IDS}" PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_TIME=08:00:00 PACK_CHAIN=1 \
    PACK_USE_MPS=0 PACK_JOB_NAME="onh_dwi_eval" \
    bash "${ROOT}/scripts/job_runner/run_job_pack_submit.sh" "${EV}"
echo "[dwi-suite] submitted. BASE=${BASE}"
