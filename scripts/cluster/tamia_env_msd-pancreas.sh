# Source AFTER .../msd-pancreas/5_scripts_msd-pancreas/00_utils/env.sh on TamIA: points msd-pancreas's test data / results at scratch and the SOURCE (pansegdata) models at ITS scratch copy.
# Override every path OUTRIGHT (a ${VAR:-} guard here is a no-op: env.sh already fired its guards). RUN_JOB_MEM_PER_GPU is in GiB (115G), never raw MB.
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/${USER}}"
export nnUNet_raw="$SCRATCH/msd-pancreas/2_nnUNet/raw"
export PREDICTIONS_ROOT="$SCRATCH/msd-pancreas/8_results/01_predictions"
export METRICS_ROOT="$SCRATCH/msd-pancreas/8_results/02_metrics"
mkdir -p "$PREDICTIONS_ROOT" "$METRICS_ROOT"
export PANSEG_DATASET_ROOT="$SCRATCH/pansegdata"
export PANSEG_NNUNET_RAW="$SCRATCH/pansegdata/2_nnUNet/raw"
export PANSEG_NNUNET_PREPROCESSED="$SCRATCH/pansegdata/2_nnUNet/preprocessed"
export PANSEG_PREDICTIONS_ROOT="$SCRATCH/pansegdata/8_results/01_predictions"
export PANSEG_DATASET_JSON="${PANSEG_NNUNET_RAW}/Dataset150_PanSegData_T1WCE/dataset.json"
export RUN_JOB_ACCOUNT="aip-jcohen"
export RUN_JOB_GPU_TYPE="h100"
