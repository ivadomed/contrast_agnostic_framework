#!/usr/bin/env bash
# Preprocess Dataset030_OnHarmonyT1w for nnUNet.
#
# Uses a single CPU slot for nnUNetv2_plan_and_preprocess.
# Must be run AFTER 01_convert_dataset.py.
# splits_final.json is already written by 01_convert_dataset.py — nnUNet will
# not regenerate it because it already exists.
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

PY=".venv/bin/python"

# nnUNet_raw, nnUNet_preprocessed, nnUNet_results set by env.sh (sourced above)

echo "[$(date '+%H:%M:%S')] Verifying dataset integrity …"
$PY -c "
import json, pathlib, os
ds = pathlib.Path(os.environ['nnUNet_raw']) / 'Dataset030_OnHarmonyT1w'
images = list((ds/'imagesTr').glob('*_0000.nii.gz'))
labels = list((ds/'labelsTr').glob('*.nii.gz'))
splits = json.load(open(ds/'splits_final.json'))
assert len(images) == len(labels), f'image/label count mismatch: {len(images)} vs {len(labels)}'
assert len(splits) == 4, f'Expected 4 folds, got {len(splits)}'
print(f'OK: {len(images)} cases, 4 folds')
"

echo "[$(date '+%H:%M:%S')] Running nnUNetv2_plan_and_preprocess …"
run_job --name on_harmony_preprocess --gpus 1 --slot 0 \
    --log "/tmp/preprocess_030.log" --wait -- bash -c "
    export nnUNet_raw='${nnUNet_raw}'
    export nnUNet_preprocessed='${nnUNet_preprocessed}'
    export nnUNet_results='${nnUNet_results}'
    cd '${PROJECT_ROOT}'
    .venv/bin/nnUNetv2_plan_and_preprocess \
        -d 030 \
        -c 3d_fullres \
        --verify_dataset_integrity \
        -np 64
"

echo "[$(date '+%H:%M:%S')] Preprocessing complete."
echo "Preprocessed data at: ${nnUNet_preprocessed}/Dataset030_OnHarmonyT1w/"
echo "Log: /tmp/preprocess_030.log"
