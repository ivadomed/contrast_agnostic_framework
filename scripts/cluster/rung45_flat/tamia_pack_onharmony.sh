#!/usr/bin/env bash
# TamIA launcher: ladder RUNG 4.5 (FLAT fill) for on-harmony (T1w, T2w, dwi_ap) x 3 folds = 9 folds on ONE whole
# 4xH100 node (3,2,2,2 per GPU), chain-resumed. Same density as the rung-6 on-harmony pack (9 folds, one node, ~34 h).
# The other 13 rung-4.5 settings train on Vulcan (rung45_flat/vulcan_launch.sh).
#
# CODE PARITY: TamIA's own AugLab checkout is older than Vulcan's and is in use by another session's packs, so it is
# NOT touched. The folds run against a PINNED exact copy of Vulcan's AugLab working tree (AugLab commit 93fb10d +
# Vulcan's working-tree files), put on PYTHONPATH. In PACK mode the recorded fold cmd does
#   export PYTHONPATH='<SCRIPTS_DIR>:${PYTHONPATH:-}'
# i.e. it appends whatever the pack job INHERITS from the submitting shell (sbatch --export=ALL), so the pin must be
# exported when submitting (done below in check/submit; CE_EXTRA_PYTHONPATH at record time does NOT reach the job).
# After start, `verify-running` reads a live trainer process's environment on its node. Repo-side training code (shared drivers, on-harmony env/trainers) is
# md5-identical to Vulcan's; nnU-Net plans/splits/dataset.json on TamIA scratch are md5-identical to Vulcan's.
#   cd /project/aip-jcohen/paulh/mri_synthesis_project
#   bash scripts/cluster/rung45_flat/tamia_pack_onharmony.sh record    # record + verify, no submit
#   bash scripts/cluster/rung45_flat/tamia_pack_onharmony.sh check     # CPU job: import resolution + transform check
#   bash scripts/cluster/rung45_flat/tamia_pack_onharmony.sh submit    # submit the chain
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
export SCRATCH="${SCRATCH:-/scratch/p/paulh}"
PIN="${SCRATCH}/rung45_flat/auglab_pin/AugLab"
CFG=transform_params_gpu_baseline_kmeans_label_remap_voronoi_flatfill_spatialDA_train050.json
STATE="${SCRATCH}/rung45_flat/PACK_DIR.txt"
MODE="${1:?usage: record|check|submit}"
O=benchmark/02_tasks/brain_healthy/on-harmony/5_scripts_on-harmony/04_train
WRAPPERS=("$O/04_62_train_t1w_baseline_kmeans_label_remap_voronoi_flatfill.sh"
          "$O/04_63_train_t2w_baseline_kmeans_label_remap_voronoi_flatfill.sh"
          "$O/04_64_train_dwi_ap_baseline_kmeans_label_remap_voronoi_flatfill.sh")
[ -f "${PIN}/auglab/transforms/gpu/palette_noisefill.py" ] || { echo "pinned AugLab missing: ${PIN}"; exit 1; }
cmp -s "${PIN}/auglab/configs/${CFG}" "sub-workspaces/auglab_workspace/AugLab/auglab/configs/${CFG}" \
  || { echo "config in the repo AugLab differs from the pin's (or is missing)"; exit 1; }

if [ -f "${STATE}" ]; then PACK="$(cat "${STATE}")"; else
  [ "${MODE}" = record ] || { echo "nothing recorded yet"; exit 1; }
  PACK="${SCRATCH}/_packruns/rung45_flat_onharmony_$(date +%Y%m%d_%H%M%S)"; mkdir -p "${SCRATCH}/rung45_flat"; echo "${PACK}" > "${STATE}"
fi
echo "[r45-tamia] PACK=${PACK}"

if [ "${MODE}" = record ]; then
  [ -f "${PACK}/index.tsv" ] && { echo "already recorded"; } || for w in "${WRAPPERS[@]}"; do
    envf="$(grep -o 'source "$(dirname "$0")/../00_utils/env[a-z0-9_]*\.sh"' "$w" | sed 's|.*/00_utils/||; s|"$||')"
    echo "[record] $(basename "$w") (env ${envf})"
    ( source "$(dirname "$w")/../00_utils/${envf}"
      source scripts/cluster/tamia_env_onharmony.sh
      export NNUNET_RESULTS_BASE="${nnUNet_results}" CE_EXTRA_PYTHONPATH="${PIN}"
      RUN_JOB_PACK_DIR="${PACK}" bash "$w" )
  done
  echo; echo "=== verification ==="; bad=0
  n=$(wc -l < "${PACK}/index.tsv"); echo "folds: $n (expect 9)"; [ "$n" = 9 ] || bad=1
  while IFS=$'\t' read -r cmd log name done; do
    clean="$(sed -e 's/\\n/\n/g' -e "s/\\\\'//g" "$cmd")"
    get() { printf '%s\n' "$clean" | sed -n "s/^ *export $1=//p" | head -1; }
    ok=OK
    [ "$(get AUGLAB_PARAMS_GPU_JSON | sed 's|.*/||')" = "${CFG}" ] || ok="${ok},WRONG_CFG"
    case "$(get PYTHONPATH)" in */5_scripts_on-harmony:'${PYTHONPATH:-}') ;; *) ok="${ok},PYTHONPATH_LINE($(get PYTHONPATH))";; esac
    case "$(get nnUNet_preprocessed)" in /scratch/*) ;; *) ok="${ok},PREPROC_NOT_SCRATCH";; esac
    case "$(get nnUNet_results)" in /scratch/*flatfill*) ;; *) ok="${ok},RESULTS($(get nnUNet_results))";; esac
    grep -q "nnUNetTrainerOnHarmonyAugLabDefault" "$cmd" || ok="${ok},TRAINER"
    [ "$ok" = OK ] || bad=1; printf '  %-6s %s\n' "$ok" "$name"
  done < "${PACK}/index.tsv"
  [ "$bad" = 0 ] && echo "VERIFY PASSED" || { echo "VERIFY FAILED"; exit 1; }
  exit 0
fi

if [ "${MODE}" = check ]; then
  # CPU job: with the RECORDED PYTHONPATH, which AugLab is imported, and is the transform the flat fill?
  c=$(head -1 "${PACK}/index.tsv" | cut -f1)
  # emulate the job: recorded line = <SCRIPTS_DIR>:${PYTHONPATH:-}, with the pin inherited from the submit shell
  PP="$(sed -e 's/\\n/\n/g' -e "s/\\\\'//g" "$c" | sed -n 's/^ *export PYTHONPATH=//p' | head -1 | sed "s|\${PYTHONPATH:-}|${PIN}|")"
  cat > "${SCRATCH}/rung45_flat/check_job.sh" <<EOF
#!/bin/bash
cd /project/aip-jcohen/paulh/mri_synthesis_project
module load python/3.11 >/dev/null 2>&1
export PYTHONPATH="${PP}" NNUNET_PROJECT_ROOT=/project/aip-jcohen/paulh/mri_synthesis_project
.venv/bin/python - <<'PY'
import hashlib, auglab, auglab.transforms.gpu.palette_noisefill as pn, auglab.transforms.gpu.transforms as tr, auglab.trainers.nnUNetTrainerDAExt as da
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
import nnunetv2, os
print("auglab path", list(auglab.__path__))
for m in (pn, tr, da):
    print(m.__name__, m.__file__, hashlib.md5(open(m.__file__,'rb').read()).hexdigest())
t = tr.AugTransformsGPU(json_path="${PIN}/auglab/configs/${CFG}")
s = [x for x in t.children() if type(x).__name__.startswith("RandomV26_6_2")][0]
print(type(s).__name__, "p", float(s.p), "sigma", s.noise_std_range, "label_fill_noise", s.label_fill_noise, "label_voronoi", s.label_voronoi)
cls = recursive_find_python_class(os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer"), "nnUNetTrainerOnHarmonyAugLabDefault", "nnunetv2.training.nnUNetTrainer")
import inspect; print("trainer", cls, inspect.getsourcefile(cls))
for b in cls.__mro__:
    if "DAExt" in b.__name__: print("  base", b.__name__, inspect.getsourcefile(b))
assert "${PIN}" in pn.__file__ and "${PIN}" in da.__file__, "pin not used"
assert list(s.noise_std_range) == [0.0, 0.0] and s.label_fill_noise is False and s.label_voronoi is False
print("CHECK PASSED")
PY
EOF
  sbatch --wait --account=aip-jcohen --time=00:20:00 --cpus-per-task=4 --mem=16000M --job-name=r45_check \
    --output="${SCRATCH}/rung45_flat/check_%j.out" "${SCRATCH}/rung45_flat/check_job.sh" \
    || sbatch --wait --partition=cpubase_bynode_b1 --account=aip-jcohen --time=00:20:00 --cpus-per-task=4 --mem=16000M \
       --job-name=r45_check --output="${SCRATCH}/rung45_flat/check_%j.out" "${SCRATCH}/rung45_flat/check_job.sh"
  cat "$(ls -t "${SCRATCH}"/rung45_flat/check_*.out | head -1)"
  exit 0
fi

if [ "${MODE}" = submit ]; then
  export PYTHONPATH="${PIN}"     # inherited by the pack jobs -> appended by every fold cmd (see header)
  PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_TIME="${PACK_TIME:-23:59:00}" PACK_CHAIN="${PACK_CHAIN:-3}" \
  PACK_USE_MPS=0 PACK_JOB_NAME="r45flat_onharmony" \
    bash scripts/job_runner/run_job_pack_submit.sh "${PACK}"
  exit 0
fi
if [ "${MODE}" = verify-running ]; then
  # (a) every live nnUNetv2_train process of each running r45 pack job: PYTHONPATH must be <SCRIPTS_DIR>:<PIN>..., and
  #     the flatfill config; (b) every started fold's saved transform_params_gpu_used_for_training.json == the config.
  bad=0
  for j in $(squeue -u "${USER}" -h -n r45flat_onharmony -t R -o %i); do
    echo "== job $j"
    out="$(srun --jobid="$j" --overlap -N1 -n1 bash -c 'for p in $(pgrep -u $USER -f "nnUNetv2_train"); do
        tr "\0" "\n" < /proc/$p/environ 2>/dev/null | grep -E "^(PYTHONPATH|AUGLAB_PARAMS_GPU_JSON)=" | tr "\n" " "; echo; done' 2>&1 | sort -u | grep -v "^$")"
    echo "${out}"
    n=$(grep -c "PYTHONPATH=" <<<"${out}" || true)
    [ "$n" -gt 0 ] || { echo "  no trainer process seen"; bad=1; }
    grep "PYTHONPATH=" <<<"${out}" | grep -vq "5_scripts_on-harmony:${PIN}" && { echo "  PIN NOT RIGHT AFTER SCRIPTS_DIR"; bad=1; }
    grep "PYTHONPATH=" <<<"${out}" | grep -vq "${CFG}" && { echo "  WRONG CONFIG IN A PROCESS"; bad=1; }
  done
  nj=0
  for f in $(find "${SCRATCH}/on-harmony/8_results/01_predictions/on_harmony_model" -path '*_flatfill_*' -name transform_params_gpu_used_for_training.json 2>/dev/null); do
    nj=$((nj+1)); cmp -s "$f" "${PIN}/auglab/configs/${CFG}" && echo "  json OK   $f" || { echo "  json DIFF $f"; bad=1; }
  done
  echo "saved configs checked: ${nj}"
  [ "$bad" = 0 ] && echo "VERIFY-RUNNING PASSED" || echo "VERIFY-RUNNING FAILED -- cancel the chain (scancel) before it burns node time"
  exit 0
fi
echo "unknown mode ${MODE}"; exit 2
