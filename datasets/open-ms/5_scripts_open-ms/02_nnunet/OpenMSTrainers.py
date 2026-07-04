"""
Registration shim — makes open-ms custom trainers discoverable by nnU-Net.

nnU-Net's recursive_find_python_class() only searches inside the installed nnunetv2
package's training/nnUNetTrainer/ dir, so this file must be copied there after every
fresh venv/nnunetv2 install:
  cp datasets/open-ms/5_scripts_open-ms/02_nnunet/OpenMSTrainers.py \
     .venv/lib/python3.*/site-packages/nnunetv2/training/nnUNetTrainer/

It imports the real implementations from the `open_ms` package
(PROJECT_ROOT/datasets/open-ms/5_scripts_open-ms/open_ms/), resolved via the
NNUNET_PROJECT_ROOT env var set by the 04_train/*.sh scripts. Method-base trainers
(nnUNetTrainerFast, AugLab's nnUNetTrainerDAExtGPU) live in src/ and the AugLab package.
"""
import os
import sys

_root = os.environ.get("NNUNET_PROJECT_ROOT", "")
if _root:
    _scripts = os.path.join(_root, "datasets", "open-ms", "5_scripts_open-ms")
    for _p in (_root, _scripts):
        if _p and _p not in sys.path:
            sys.path.insert(0, _p)

try:
    from open_ms.trainers.baseline import nnUNetTrainerOpenMSBaseline            # noqa: F401
    from open_ms.trainers.auglab_default import nnUNetTrainerOpenMSAugLabDefault  # noqa: F401
    from open_ms.trainers.auglab_valsynth import nnUNetTrainerOpenMSAugLabValSynth  # noqa: F401
    from open_ms.trainers.v26_6_2 import nnUNetTrainerOpenMSV26_6_2              # noqa: F401
except Exception as _e:  # pragma: no cover — discovery must not hard-crash nnU-Net import
    print(f"[OpenMSTrainers] import failed: {_e}")
