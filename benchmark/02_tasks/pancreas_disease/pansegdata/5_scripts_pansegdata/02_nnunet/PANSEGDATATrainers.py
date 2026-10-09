"""
Registration shim — makes pansegdata custom trainers discoverable by nnU-Net.

nnU-Net's recursive_find_python_class() only searches inside the installed nnunetv2
package's training/nnUNetTrainer/ dir, so this file must be copied there after every
fresh venv/nnunetv2 install, ON EVERY CLUSTER (Vulcan and TamIA separately):
  cp benchmark/02_tasks/pancreas_disease/pansegdata/5_scripts_pansegdata/02_nnunet/PANSEGDATATrainers.py \
     .venv/lib/python3.*/site-packages/nnunetv2/training/nnUNetTrainer/
(The venv-setup script auto-discovers *Trainers.py under benchmark/02_tasks; see the project notes.)

It imports the real implementations from the `pansegdata` package
(PROJECT_ROOT/benchmark/02_tasks/pancreas_disease/pansegdata/5_scripts_pansegdata/), resolved via the
NNUNET_PROJECT_ROOT env var set by the 04_train/*.sh scripts.
"""
import os
import sys

_root = os.environ.get("NNUNET_PROJECT_ROOT", "")
if _root:
    _scripts = os.path.join(_root, "benchmark", "02_tasks", "pancreas_disease", "pansegdata", "5_scripts_pansegdata")
    for _p in (_root, _scripts):
        if _p and _p not in sys.path:
            sys.path.insert(0, _p)

try:
    from pansegdata.trainers.baseline import nnUNetTrainerPANSEGDATABaseline              # noqa: F401
    from pansegdata.trainers.auglab_default import nnUNetTrainerPANSEGDATAAugLabDefault    # noqa: F401
    from pansegdata.trainers.auglab_valsynth import nnUNetTrainerPANSEGDATAAugLabValSynth  # noqa: F401
    from pansegdata.trainers.auglab_dualval import nnUNetTrainerPANSEGDATAAugLabDualVal    # noqa: F401
except Exception as _e:  # pragma: no cover — discovery must not hard-crash nnU-Net import
    print(f"[PANSEGDATATrainers] import failed: {_e}")
