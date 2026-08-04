#!/usr/bin/env bash
# Node-pack runner for the label boundary-cue analysis -- ALL branches of ALL datasets inside ONE
# job, spread across the node's GPUs.
#
# Why this exists rather than calling each dataset's run_*.sh: on TamIA allocation is WHOLE-NODE
# (4x H100), so submitting 8 one-GPU branches as 8 jobs would idle 3 GPUs per job for the whole
# wall-clock duration -- exactly the under-utilisation Alliance staff have already warned this
# account about (CLAUDE.md). This packs every branch onto the node at PACK_PER_GPU per GPU.
#
# Runs the branches listed in a branch file, one line per branch:
#   <dataset>|<modality>|<imagesDir>|<labelsDir>|<dataset.json>|<imageSuffix>|<labelSuffix>|<unionIds>|<unionName>
#
# Usage (from a login node; submits and returns):
#   BRANCH_FILE=/path/to/branches.txt OUT_ROOT=/path N_SUBJECTS=5 bash run_label_cues_pack.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../../.." && pwd)"
PY="${REPO}/.venv/bin/python"
: "${BRANCH_FILE:?set BRANCH_FILE}"
: "${OUT_ROOT:?set OUT_ROOT}"
: "${N_SUBJECTS:=40}"
: "${PACK_PER_GPU:=2}"
: "${N_GPUS:=4}"
: "${JOB_TIME:=03:00:00}"
: "${JOB_NAME:=cue_pack}"
LOGDIR="${OUT_ROOT}/logs"
mkdir -p "${OUT_ROOT}" "${LOGDIR}"

# NOTE: the phantom self-test runs INSIDE the job (first thing the worker does), not here. It is
# still a hard precondition -- the worker aborts the whole pack if it fails -- but it must not run
# on the login node: on TamIA's login node it took ~2 MINUTES PER PHANTOM (~10 CPU-min total,
# right at the Alliance login-node allowance) versus ~1 s on a compute node, and a submission was
# already lost to a timeout partway through it. On the GPU it is seconds.

WORK="${OUT_ROOT}/_pack_worker.sh"
cat > "${WORK}" <<'EOS'
#!/usr/bin/env bash
set -uo pipefail
PY="$1"; HERE="$2"; BRANCH_FILE="$3"; OUT_ROOT="$4"; N_SUBJECTS="$5"; PACK_PER_GPU="$6"; N_GPUS="$7"
SLOTS=$(( N_GPUS * PACK_PER_GPU ))

# Hard precondition, on the compute node: the phantom self-test is what establishes that the four
# cues are separable at all. No branch runs if it fails.
if ! "${PY}" "${HERE}/cue_metrics.py" --sanity > "${OUT_ROOT}/logs/sanity.log" 2>&1; then
    echo "SANITY FAILED -- aborting pack, no results produced. See ${OUT_ROOT}/logs/sanity.log"
    exit 1
fi
echo "sanity passed"
i=0
while IFS='|' read -r DS MOD IMAGES LABELS DSJSON ISUF LSUF UIDS UNAME; do
    [ -z "${DS:-}" ] && continue
    GPU=$(( (i % SLOTS) / PACK_PER_GPU ))
    extra=""
    [ -n "${UIDS:-}" ] && extra="--extra-union ${UIDS} --union-name ${UNAME}"
    # NOTE: ${extra} MUST be word-split here (hence unquoted) -- it carries --extra-union/--union-name.
    # The pilot run silently produced NO whole-tumour rows because this variable was built and then
    # never passed; the job succeeded and the omission was only visible as a missing table row.
    CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" "${HERE}/compute_label_cues.py" \
        --images "${IMAGES}" --labels "${LABELS}" --labels-json "${DSJSON}" \
        --image-suffix "${ISUF}" --label-suffix "${LSUF}" \
        --out-dir "${OUT_ROOT}/data" --dataset "${DS}" --modality "${MOD}" \
        --n-subjects "${N_SUBJECTS}" --device cuda ${extra} \
        > "${OUT_ROOT}/logs/${DS}__${MOD}.log" 2>&1 &
    echo "launched ${DS}/${MOD} on GPU ${GPU}"
    i=$(( i + 1 ))
    # Keep at most SLOTS processes resident; wait for one to finish before launching more.
    while [ "$(jobs -rp | wc -l)" -ge "${SLOTS}" ]; do sleep 5; done
done < "${BRANCH_FILE}"
wait
echo "PACK_DONE"
EOS
chmod +x "${WORK}"

source "${REPO}/scripts/job_runner/run_job.sh"
run_job --name "${JOB_NAME}" --gpus "${N_GPUS}" --time "${JOB_TIME}" \
        --log "${LOGDIR}/pack.log" -- \
    bash "${WORK}" "${PY}" "${HERE}" "${BRANCH_FILE}" "${OUT_ROOT}" "${N_SUBJECTS}" \
         "${PACK_PER_GPU}" "${N_GPUS}"
echo "submitted ${JOB_NAME}: $(wc -l < "${BRANCH_FILE}") branches over ${N_GPUS} GPUs"
