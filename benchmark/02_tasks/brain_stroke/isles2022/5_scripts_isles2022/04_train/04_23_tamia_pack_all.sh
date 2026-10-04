#!/usr/bin/env bash
# TamIA whole-node pack LAUNCHER for ALL isles2022 training: 10 runs per contrast (6 headline methods -- OURS is ONE DualVal run that yields
# BOTH the val000 and val100 mirrors -- plus ladder rungs 2-5) x 2 contrasts x folds 0 1 2 = 60 fold-jobs, split at FOLD granularity over
# several whole-node pack chains, load-balanced by per-method cost (LPT: heaviest fold first onto the least-loaded node/GPU).
#
# Run ON TamIA, from the repo root:   bash <this file> [--dry-run]
# Env (defaults = the measured plan; see 04_24_tamia_sizing_probe.sh + 04_25_probe_report.sh -- do NOT launch without a probe):
#   FOLDS_PER_PACK   folds packed on one node (default 12 = 3 per H100)       PACK_TIME   per-job walltime (default 23:59:00, < the 24h cap)
#   PACK_CHAIN       chained jobs per pack, each resuming the last's checkpoints (default 3; must satisfy chain x 23.9h >= 2000 epochs x s/epoch)
#   COST_<method>    relative per-fold cost for load balancing (defaults below; set from the probe's contended s/epoch)
#   NNUNET_NUM_EPOCHS 2000 (set by the wrappers; Paul 2026-10-04)
# Fixed RUN_IDs are persisted in $PACKROOT/RUN_IDS.tsv (first invocation stamps the timestamp) and the pack dirs are persistent, so
# re-running this script resumes/extends (train_common.sh pack mode resumes from checkpoint_latest at RUNTIME). PACK_DIR/RUN_IDs name the
# contrast, so the two contrasts can never overwrite each other's recorded commands.
set -euo pipefail
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../../../../.." && pwd)"
export SCRATCH="${SCRATCH:-/scratch/p/paulh}"
export RUN_JOB_CPUS_PER_GPU=12 RUN_JOB_MEM_PER_GPU=115G        # whole-node H100: 48 CPUs / 4 GPUs; GiB, never raw MB
FOLDS_PER_PACK="${FOLDS_PER_PACK:-12}"; PACK_CHAIN="${PACK_CHAIN:-3}"; PACK_TIME="${PACK_TIME:-23:59:00}"
PACKROOT="${PACKROOT:-${SCRATCH}/isles2022/_packruns}"; mkdir -p "${PACKROOT}"
[ "${DRY}" = 1 ] || { source "${HERE}/../00_utils/env.sh"; source "${ROOT}/scripts/cluster/tamia_env_isles2022.sh"; }

# contrast | wrapper | default relative cost (baseline = 1; refine from the probe)
RUNS=(
 "dwi|04_01_train_dwi_baseline.sh|${COST_baseline:-1}"
 "dwi|04_02_train_dwi_auglab_default.sh|${COST_auglab_default:-1}"
 "dwi|04_03_train_dwi_synthseg_noEM.sh|${COST_synthseg_noEM:-1}"
 "dwi|04_04_train_dwi_synthseg_EM.sh|${COST_synthseg_EM:-1}"
 "dwi|04_05_train_dwi_srcsm.sh|${COST_srcsm:-1}"
 "dwi|04_06_train_dwi_auglabAug_v26_6_2_dualval.sh|${COST_dualval:-1}"
 "dwi|04_15_train_dwi_baseline_kmeans.sh|${COST_kmeans:-1}"
 "dwi|04_16_train_dwi_baseline_kmeans_label_remap.sh|${COST_kmeans:-1}"
 "dwi|04_17_train_dwi_baseline_kmeans_label_remap_voronoi.sh|${COST_kmeans:-1}"
 "dwi|04_18_train_dwi_v26_6_2_train050_val100.sh|${COST_v26alone:-1}"
 "flair|04_08_train_flair_baseline.sh|${COST_baseline:-1}"
 "flair|04_09_train_flair_auglab_default.sh|${COST_auglab_default:-1}"
 "flair|04_10_train_flair_synthseg_noEM.sh|${COST_synthseg_noEM:-1}"
 "flair|04_11_train_flair_synthseg_EM.sh|${COST_synthseg_EM:-1}"
 "flair|04_12_train_flair_srcsm.sh|${COST_srcsm:-1}"
 "flair|04_13_train_flair_auglabAug_v26_6_2_dualval.sh|${COST_dualval:-1}"
 "flair|04_19_train_flair_baseline_kmeans.sh|${COST_kmeans:-1}"
 "flair|04_20_train_flair_baseline_kmeans_label_remap.sh|${COST_kmeans:-1}"
 "flair|04_21_train_flair_baseline_kmeans_label_remap_voronoi.sh|${COST_kmeans:-1}"
 "flair|04_22_train_flair_v26_6_2_train050_val100.sh|${COST_v26alone:-1}"
)
RUNIDS="${PACKROOT}/RUN_IDS.tsv"
if [ ! -f "${RUNIDS}" ] || [ "${DRY}" = 1 ]; then TS="$(date +%Y%m%d_%H%M%S)"; else TS=""; fi
items="$(mktemp)"; ids_new="$(mktemp)"
for r in "${RUNS[@]}"; do
    IFS='|' read -r C W COST <<< "$r"
    M="$(grep -m1 -E '^METHOD="' "${HERE}/${W}" | sed -E 's/^METHOD="([^"]*)".*/\1/')"
    if [ -f "${RUNIDS}" ] && grep -q -P "^${C}\t${M}\t" "${RUNIDS}"; then RID="$(awk -F'\t' -v c="$C" -v m="$M" '$1==c && $2==m {print $3}' "${RUNIDS}")"
    else RID="isles2022_${C}_${M}_${TS:-$(date +%Y%m%d_%H%M%S)}"; printf '%s\t%s\t%s\n' "$C" "$M" "$RID" >> "${ids_new}"; fi
    for F in 0 1 2; do printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${COST}" "$C" "$M" "$W" "$RID" "$F" >> "${items}"; done
done
[ "${DRY}" = 1 ] || { if [ -s "${ids_new}" ]; then cat "${ids_new}" >> "${RUNIDS}"; fi; }

# LPT placement: sort by cost desc; each item -> the pack with the lowest load that still has room (PACKS = ceil(items / FOLDS_PER_PACK)),
# then within the pack -> the GPU with the lowest load (4 GPUs). Output: pack gpu cost C M W RID F
NITEMS=$(wc -l < "${items}"); PACKS=$(( (NITEMS + FOLDS_PER_PACK - 1) / FOLDS_PER_PACK ))
sort -t$'\t' -k1,1nr "${items}" | awk -F'\t' -v P="${PACKS}" -v CAP="${FOLDS_PER_PACK}" -v G=4 '
  { best=-1; for (p=1;p<=P;p++) if (cnt[p]<CAP && (best<0 || load[p]<load[best])) best=p;
    bg=0; for (g=1;g<G;g++) if (gl[best,g]<gl[best,bg]) bg=g;
    cnt[best]++; load[best]+=$1; gl[best,bg]+=$1; printf "%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\n", best, bg, $1,$2,$3,$4,$5,$6 }' \
  | sort -t$'\t' -k1,1n -k2,2n > "${items}.plan"
echo "[pack] ${NITEMS} fold-jobs -> ${PACKS} whole-node packs (<=${FOLDS_PER_PACK} folds each), chain ${PACK_CHAIN} x ${PACK_TIME}"
for p in $(seq 1 "${PACKS}"); do
    echo "  pack ${p}: $(awk -F'\t' -v p=$p '$1==p {n++; l+=$3} END {printf "%d folds, load %.1f", n, l}' "${items}.plan")  GPU map: $(awk -F'\t' -v p=$p '$1==p {printf "%s ", $2}' "${items}.plan")"
    awk -F'\t' -v p=$p '$1==p {printf "      GPU%s  %-8s %-42s fold%s (cost %s)\n", $2,$4,$5,$8,$3}' "${items}.plan"
done
[ "${DRY}" = 1 ] && { echo "[pack] dry run: nothing recorded or submitted"; exit 0; }

for p in $(seq 1 "${PACKS}"); do
    PD="${PACKROOT}/pack${p}"; mkdir -p "${PD}"
    if [ -f "${PD}/index.tsv" ]; then echo "[pack] pack${p}: index.tsv exists -> reusing recording (resume/extend)"; else
        while IFS=$'\t' read -r pk gpu cost C M W RID F; do
            [ "$pk" = "$p" ] || continue
            ( [ "${C}" = flair ] && source "${HERE}/../00_utils/env_flair.sh"
              RUN_JOB_PACK_DIR="${PD}" TRAIN_FOLDS="${F}" bash "${HERE}/${W}" "${RID}" < /dev/null )
        done < "${items}.plan"
        N=$(grep -c . "${PD}/index.tsv"); E=$(awk -F'\t' -v p=$p '$1==p' "${items}.plan" | wc -l)
        [ "${N}" = "${E}" ] || { echo "ERROR: pack${p} recorded ${N} fold cmds, expected ${E}" >&2; exit 1; }
        # contrast sanity: no recorded cmd may reference both contrasts' result dirs
        for f in "${PD}"/*.sh; do case "$f" in *_packjob_body.sh) continue;; esac
            grep -q "isles2022_model/dwi/" "$f" && grep -q "isles2022_model/flair/" "$f" && { echo "ERROR: $f mixes contrasts" >&2; exit 1; } || true
        done
    fi
    # PACK_GPU_MAP must follow index.tsv ROW order (= recording order = plan order, sorted by gpu within the pack)
    GMAP="$(awk -F'\t' -v p=$p '$1==p {printf "%s ", $2}' "${items}.plan")"
    PACK_GPU_MAP="${GMAP}" PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_TIME="${PACK_TIME}" PACK_CHAIN="${PACK_CHAIN}" \
      PACK_USE_MPS="${PACK_USE_MPS:-0}" PACK_JOB_NAME="isles2022_pack${p}" \
      bash "${ROOT}/scripts/job_runner/run_job_pack_submit.sh" "${PD}"
done
echo "[pack] submitted ${PACKS} chains. RUN_IDs: ${RUNIDS}. Progress lives in each run's fold_N/training_log_*.txt under ${PREDICTIONS_ROOT:-\$PREDICTIONS_ROOT}."
