#!/bin/bash
# CPU check (run through run_job, never on the login node): every generated val000 real-fill wrapper's TRAINER class is
# discoverable by nnU-Net in this checkout's venv, and its config JSONs exist. Usage: bash check_trainers.sh
set -uo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
module load python/3.11 >/dev/null 2>&1 || true
export NNUNET_PROJECT_ROOT="$PWD"
rc=0
for w in $(ls benchmark/02_tasks/*/*/5_scripts_*/04_train/04_*_v26_6_2*_train050_val000.sh | grep -v auglabAug); do
  t=$(grep -m1 -E '^TRAINER=' "$w" | sed -E 's/^TRAINER="([^"]*)".*/\1/')
  .venv/bin/python - "$t" <<'PY' || { echo "FAIL trainer $t ($w)"; rc=1; }
import sys, warnings
warnings.simplefilter("error", UserWarning)   # a shim's ImportError is only a warning -- make it fatal here
import nnunetv2
from os.path import join
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
c = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"), sys.argv[1], "nnunetv2.training.nnUNetTrainer")
assert c is not None, sys.argv[1]
print("ok", sys.argv[1], c.__module__)
PY
  for j in $(grep -oE 'transform_params_gpu_[A-Za-z0-9_]+\.json' "$w" | sort -u); do
    [ -f "sub-workspaces/auglab_workspace/AugLab/auglab/configs/$j" ] || { echo "FAIL config $j ($w)"; rc=1; }
  done
done
echo "check_trainers rc=$rc"; exit $rc
