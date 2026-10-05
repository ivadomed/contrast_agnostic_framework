#!/usr/bin/env bash
# KILLARNEY LAUNCH of ALL pansegdata training (run ON Killarney, after 04_26/04_27 sizing probes; user instruction 2026-10-04: all compute on Killarney).
# Killarney allocates per GPU, so every fold is its own single-GPU sbatch job through run_job (the shared 04_XX wrappers -> 04_00_common.sh -> train_common.sh):
#   2 contrasts x 10 runs (6 headline methods incl. OURS DualVal [one run -> val000 + val100 mirror] + 4 ladder rungs) x folds 0 1 2 = 60 fold jobs, 2000 epochs each.
# GPU class: H100 (probe 2026-10-04: ~2.2x faster than L40S; 2000-epoch single-fold wall, H100 / L40S hours: baseline 4.9/12.6, auglab_default 6.2-7.6/14.5,
# srcsm 6.4-6.8/13.8, synthseg_noEM 6.6-7.4/14.7, synthseg_EM 8.0-8.2/17.8, OURS DualVal 16.8-17.1/29-31) -> only H100 fits the slowest method in one <=24 h job.
# Job time limit per method = ~1.4x the probe wall (rungs 2-4 were NOT probed: 16 h guess ~ synthseg_noEM..EM cost x2); a job that hits its limit is resumable:
# re-run its wrapper with the SAME RUN_ID (restarts from checkpoint_latest).
#   bash 04_28_killarney_launch_all.sh --dry-run     # print the plan only
#   bash 04_28_killarney_launch_all.sh               # submit all 60 (spaced)
#   LAUNCH_GPU_TYPE=l40s bash 04_28_killarney_launch_all.sh   # fallback class (then raise the OURS / rung time limits: ~31 h, needs the <=3 d tier)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
export RUN_JOB_ACCOUNT="${RUN_JOB_ACCOUNT:-aip-jcohen}" RUN_JOB_GPU_TYPE="${LAUNCH_GPU_TYPE:-h100}"
declare -A TLIM=( [baseline]=08:00:00 [auglab_default]=11:00:00 [srcsm]=11:00:00 [synthseg_noEM]=11:00:00 [synthseg_EM]=11:50:00
                  [baseline_kmeans]=16:00:00 [baseline_kmeans_label_remap]=16:00:00 [baseline_kmeans_label_remap_voronoi]=16:00:00
                  [v26_6_2_train050_val100]=23:30:00 [auglabAug_v26_6_2_train050_val000]=23:30:00 )
WRAPPERS=( "${HERE}"/04_0[1-6]_train_t1wce_*.sh "${HERE}"/04_1[5-8]_train_t1wce_*.sh "${HERE}"/04_{08,09,10,11,12,13}_train_t2w_*.sh "${HERE}"/04_{19,20,21,22}_train_t2w_*.sh )
[ "${#WRAPPERS[@]}" = 20 ] || { echo "ERROR: expected 20 training wrappers, found ${#WRAPPERS[@]}" >&2; exit 1; }
echo "[launch] gpu=${RUN_JOB_GPU_TYPE} account=${RUN_JOB_ACCOUNT} wrappers=${#WRAPPERS[@]} (x3 folds = $(( ${#WRAPPERS[@]} * 3 )) jobs) dry_run=${DRY}"
for w in "${WRAPPERS[@]}"; do
    M="$(grep -m1 -E '^METHOD="' "$w" | sed -E 's/^METHOD="([^"]*)".*/\1/')"
    C="$(basename "$w" | sed -E 's/^04_[0-9]+_train_(t1wce|t2w)_.*/\1/')"
    T="${TLIM[$M]:-}"; [ -n "$T" ] || { echo "ERROR: no time limit defined for method '$M' (${w##*/})" >&2; exit 1; }
    printf "[launch] %-6s %-40s time=%s  (%s)\n" "$C" "$M" "$T" "${w##*/}"
    [ "$DRY" = 1 ] && continue
    ( export RUN_JOB_TIME_DEFAULT="$T"; bash "$w" ) < /dev/null
    sleep 3    # space out sbatch submissions
done
[ "$DRY" = 1 ] || echo "[launch] done. Watch: bash ${HERE}/04_29_killarney_status.sh"
