#!/usr/bin/env bash
# TAMIA whole-node PREDICT job, chained via a NATIVE Slurm --dependency off the training
# pack chain (04_33) — NOT a polling/monitoring script. Predict runs entirely on tamia:
# checkpoints AND test inputs (imagesTs_t1n/t1c/t2w/t2f) are already staged there.
#
# Unlike training, this canNOT use run_job_pack_submit.sh's usual "record outside, then
# submit a job that replays the pre-built index.tsv" flow: the DualVal val100 MIRROR run
# dir (brats2024-glioma_t2w_..._val100_...) is only materialized by on_train_end, so it
# doesn't exist until training is actually finished — i.e. exactly when the Slurm
# dependency below resolves. So THIS script submits ONE self-contained whole-node job
# that, once the dependency fires, itself: (1) safety-checks all 6 folds have
# checkpoint_final, (2) RECORDS the three predict wrappers' fold commands (via
# RUN_JOB_PACK_DIR, same generic run_job pack-record mechanism as training — cheap,
# no compute) now that every run dir is guaranteed to exist, (3) launches all recorded
# fold-commands round-robin across the node's 4 GPUs, waits, done.
#
# Usage (run ON tamia):
#   bash 05_25_tamia_pack_predict_after_training.sh <dependency, e.g. afterany:383046>
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"

DEPENDENCY="${1:?usage: 05_25_tamia_pack_predict_after_training.sh <afterany:jobid>}"

T2W_VAL000="brats2024-glioma_t2w_auglabAug_v26_6_2_train050_val000_20260725_113540"
T2W_VAL100="brats2024-glioma_t2w_auglabAug_v26_6_2_train050_val100_20260725_113540"   # auto-materialized DualVal mirror
T1N_VAL100="brats2024-glioma_t1n_auglabAug_v26_6_2_train050_val100_20260725_113540"

PACK_DIR="/scratch/p/paulh/brats2024-glioma/_packruns/predict_pack_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${PACK_DIR}"
echo "${PACK_DIR}" > /scratch/p/paulh/brats2024-glioma/_packruns/.last_predict_pack_dir

BODY="${PACK_DIR}/_predict_job_body.sh"
cat > "${BODY}" <<SBODY
#!/bin/bash
#SBATCH --job-name=brats_pack_predict
#SBATCH --account=aip-jcohen
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --output=${PACK_DIR}/predict_job_%j.out
set -uo pipefail
cd '${HERE}'

B="/scratch/p/paulh/brats2024-glioma/8_results/01_predictions/brats2024_glioma_model"
missing=0
for k in 0 1 2; do
    [ -f "\${B}/t2w/auglab/${T2W_VAL000}/Dataset052_BraTS2024GliomaT2w/nnUNetTrainerBraTS2024GliomaT2wAugLabDualVal__nnUNetPlans__3d_fullres/fold_\${k}/checkpoint_final.pth" ] || missing=1
    [ -f "\${B}/t1n/auglab/${T1N_VAL100}/Dataset051_BraTS2024GliomaT1n/nnUNetTrainerBraTS2024GliomaAugLabValSynth__nnUNetPlans__3d_fullres/fold_\${k}/checkpoint_final.pth" ] || missing=1
done
if [ "\${missing}" = "1" ]; then
    echo "[predict-job] ERROR: training not actually complete (missing checkpoint_final) — refusing to predict. Extend the training chain and resubmit." >&2
    exit 1
fi
echo "[predict-job] training verified complete — recording predict commands"

RUN_JOB_PACK_DIR='${PACK_DIR}' bash 05_22_predict_t2w_auglabAug_v26_6_2_train050_val000_dualval.sh '${T2W_VAL000}'
RUN_JOB_PACK_DIR='${PACK_DIR}' bash 05_23_predict_t2w_auglabAug_v26_6_2_train050_val100_dualval.sh '${T2W_VAL100}'
RUN_JOB_PACK_DIR='${PACK_DIR}' bash 05_24_predict_t1n_auglabAug_v26_6_2_train050_val100.sh          '${T1N_VAL100}'

echo "[predict-job] recorded fold-commands:"
cut -f3 '${PACK_DIR}/index.tsv' | sed 's/^/  /'

echo "[predict-job] host=\$(hostname) job=\$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

declare -a PIDS NAMES GPUS
i=0
G=4
while IFS=\$'\t' read -r cmdfile log name donefile; do
    [ -n "\$cmdfile" ] || continue
    gpu=\$(( i % G ))
    plog="\${cmdfile%.sh}.log"
    echo "[predict-job] launch '\$name' on GPU \$gpu -> \$plog"
    CUDA_VISIBLE_DEVICES=\$gpu bash "\$cmdfile" > "\$plog" 2>&1 &
    PIDS[\$i]=\$!; NAMES[\$i]="\$name"; GPUS[\$i]=\$gpu
    i=\$(( i + 1 ))
    sleep 5
done < '${PACK_DIR}/index.tsv'

echo "[predict-job] \$i fold-predict commands launched; waiting..."
rc_all=0
for j in "\${!PIDS[@]}"; do
    if wait "\${PIDS[\$j]}"; then echo "[predict-job] OK   '\${NAMES[\$j]}' (GPU \${GPUS[\$j]})"
    else rc=\$?; rc_all=1; echo "[predict-job] FAIL '\${NAMES[\$j]}' (GPU \${GPUS[\$j]}) exit=\$rc"; fi
done
echo "[predict-job] done rc_all=\$rc_all"
exit \$rc_all
SBODY

echo "[tamia-predict-pack] submitting predict job, --dependency=${DEPENDENCY}"
sbatch --dependency="${DEPENDENCY}" "${BODY}"
