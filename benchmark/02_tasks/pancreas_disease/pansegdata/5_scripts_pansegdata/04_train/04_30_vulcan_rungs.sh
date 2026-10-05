#!/usr/bin/env bash
# VULCAN run of the pansegdata ladder rungs 2-5 (baseline_kmeans / +label_remap / +voronoi / "v26 alone") x 2 contrasts x 3 folds = 24 single-GPU L40S jobs.
# Why (2026-10-05): Killarney's H100 queue left these 24 folds pending for 15+ h with nothing running (several Killarney nodes down/draining); Vulcan had idle L40S nodes.
# Run ON Vulcan (this checkout). Needs: 2_nnUNet_pansegdata/{raw,preprocessed} rsynced from Killarney, the trainer shim in Vulcan's venv, and the PINNED AugLab copy.
# CODE PARITY: the headline runs trained on Killarney with AugLab 7b761b5; Vulcan's own AugLab has 2 extra code commits (opt-in boundary partial-volume) + uncommitted configs. So the rungs run
# against an exact copy of Killarney's AugLab put first on PYTHONPATH (CE_EXTRA_PYTHONPATH -> common_env.sh). The 5 config JSONs the rungs use are byte-identical on both sides (md5-checked);
# venv trainer copy (nnUNetTrainerDAExt.py), nnunet_logger.py and torch/nnunetv2/kornia/batchgenerators/numpy/blosc2 versions are identical too.
#   bash 04_30_vulcan_rungs.sh --dry-run
#   bash 04_30_vulcan_rungs.sh --probe                          # fold 0 of the 4 t1wce rungs, 20 epochs, throwaway base: measures s/epoch (rungs were never probed)
#   TL_KMEANS=<hh:mm:ss> TL_V26=<hh:mm:ss> bash 04_30_vulcan_rungs.sh --launch    # time limits are REQUIRED (set them from the probe), never guessed here
# A job that hits its limit resumes by re-running its wrapper with the SAME RUN_ID. Results land in this checkout's 8_results_pansegdata/ (RUN_IDs: _launch/RUN_IDS_vulcan_rungs.tsv).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:---dry-run}"
export RUN_JOB_ACCOUNT="${RUN_JOB_ACCOUNT:-aip-jcohen}" RUN_JOB_GPU_TYPE="${LAUNCH_GPU_TYPE:-l40s}"
export CE_EXTRA_PYTHONPATH="${PIN_AUGLAB:-/project/aip-jcohen/paulh/pansegdata_auglab_7b761b5/AugLab}"
[ -d "${CE_EXTRA_PYTHONPATH}/auglab" ] || { echo "ERROR: pinned AugLab copy missing at ${CE_EXTRA_PYTHONPATH}" >&2; exit 1; }
T1=( "${HERE}"/04_1[5-8]_train_t1wce_*.sh ); T2=( "${HERE}"/04_{19,20,21,22}_train_t2w_*.sh )
[ "${#T1[@]}" = 4 ] && [ "${#T2[@]}" = 4 ] || { echo "ERROR: expected 4+4 rung wrappers, got ${#T1[@]}+${#T2[@]}" >&2; exit 1; }
method_of() { grep -m1 -E '^METHOD="' "$1" | sed -E 's/^METHOD="([^"]*)".*/\1/'; }
case "${MODE}" in
  --probe)
    PROBE="${SCRATCH:?}/pansegdata/_probe_vulcan_rungs"; export RESULTS_DIR="${PROBE}/results" PREDICTIONS_ROOT="${PROBE}/01_predictions"
    mkdir -p "${PREDICTIONS_ROOT}"; export TRAIN_FOLDS=0 NNUNET_NUM_EPOCHS="${PROBE_EPOCHS:-20}" RUN_JOB_TIME_DEFAULT="${PROBE_TIME:-01:30:00}"
    for w in "${T1[@]}"; do M="$(method_of "$w")"; echo "[probe] ${M}"; ( bash "$w" "pansegdata_t1wce_${M}_PROBEV" ) < /dev/null; sleep 2; done
    echo "[probe] submitted 4 jobs. Report: bash ${HERE}/04_27_probe_report.sh ${PROBE}" ;;
  --dry-run|--launch)
    if [ "${MODE}" = "--launch" ]; then
      : "${TL_KMEANS:?set TL_KMEANS (time limit for the kmeans / label_remap / voronoi rungs) from the probe}" "${TL_V26:?set TL_V26 (time limit for the v26-alone rung) from the probe}"
      LOGF="${HERE}/../../8_results_pansegdata/_launch/vulcan_rungs_launch.log"; mkdir -p "$(dirname "${LOGF}")"
    fi
    for w in "${T1[@]}" "${T2[@]}"; do
      M="$(method_of "$w")"; C="$(basename "$w" | sed -E 's/^04_[0-9]+_train_(t1wce|t2w)_.*/\1/')"
      if [ "$M" = "v26_6_2_train050_val100" ]; then T="${TL_V26:-<TL_V26>}"; else T="${TL_KMEANS:-<TL_KMEANS>}"; fi
      printf "[vulcan] %-6s %-40s time=%s gpu=%s (%s)\n" "$C" "$M" "$T" "${RUN_JOB_GPU_TYPE}" "${w##*/}"
      [ "${MODE}" = "--dry-run" ] && continue
      ( export RUN_JOB_TIME_DEFAULT="$T"; bash "$w" ) < /dev/null 2>&1 | tee -a "${LOGF}"
      sleep 3
    done
    [ "${MODE}" = "--dry-run" ] || echo "[vulcan] launched. Status: bash ${HERE}/04_29_killarney_status.sh   (works on any checkout)" ;;
  *) echo "usage: $0 --dry-run | --probe | --launch" >&2; exit 2 ;;
esac
