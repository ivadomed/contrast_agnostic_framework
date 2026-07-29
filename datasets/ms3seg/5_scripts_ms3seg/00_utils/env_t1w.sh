#!/usr/bin/env bash
# Override OPENMS_* env vars for T1w-TRAINED open-ms models. Source instead of env.sh.
export OPENMS_TRAINING_CONTRAST="t1w"
export OPENMS_DATASET_ID="71"
export OPENMS_DS_NAME="Dataset071_OpenMS_T1W"
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
