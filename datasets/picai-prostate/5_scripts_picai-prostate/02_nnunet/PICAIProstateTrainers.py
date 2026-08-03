"""
Registration shim — makes picai-prostate custom trainers discoverable by nnU-Net.

nnU-Net's recursive_find_python_class() only searches inside the installed nnunetv2
package's training/nnUNetTrainer/ dir, so this file must be copied there after every
fresh venv/nnunetv2 install:
  cp datasets/picai-prostate/5_scripts_picai-prostate/02_nnunet/PICAIProstateTrainers.py \
     .venv/lib/python3.*/site-packages/nnunetv2/training/nnUNetTrainer/

It imports the real implementations from the `picai_prostate` package
(PROJECT_ROOT/datasets/picai-prostate/5_scripts_picai-prostate/picai_prostate/), resolved via the
NNUNET_PROJECT_ROOT env var set by the 04_train/*.sh scripts. Method-base trainers
(nnUNetTrainerFast, AugLab's nnUNetTrainerDAExtGPU) live in src/ and the AugLab package.
"""
import os
import sys

_root = os.environ.get("NNUNET_PROJECT_ROOT", "")
if _root:
    _scripts = os.path.join(_root, "datasets", "picai-prostate", "5_scripts_picai-prostate")
    for _p in (_root, _scripts):
        if _p and _p not in sys.path:
            sys.path.insert(0, _p)

try:
    from picai_prostate.trainers.baseline import nnUNetTrainerPICAIProstateBaseline            # noqa: F401
    from picai_prostate.trainers.auglab_default import nnUNetTrainerPICAIProstateAugLabDefault  # noqa: F401
    from picai_prostate.trainers.auglab_valsynth import nnUNetTrainerPICAIProstateAugLabValSynth  # noqa: F401
    from picai_prostate.trainers.v26_6_2 import nnUNetTrainerPICAIProstateV26_6_2              # noqa: F401
    from picai_prostate.trainers.auglab_dualval import nnUNetTrainerPICAIProstateAugLabDualVal  # noqa: F401  # opt-in only, see module docstring
except Exception as _e:  # pragma: no cover — discovery must not hard-crash nnU-Net import
    print(f"[PICAIProstateTrainers] import failed: {_e}")
