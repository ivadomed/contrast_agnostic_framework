"""
Registration shim — makes autopet custom trainers discoverable by nnU-Net.

nnU-Net's recursive_find_python_class() only searches inside the installed nnunetv2
package's training/nnUNetTrainer/ dir, so this file must be copied there after every
fresh venv/nnunetv2 install, ON EVERY CLUSTER (Vulcan and TamIA separately — see
CLAUDE.md's "registration shims" section; same mechanism as BraTS2024GliomaTrainers.py /
CHAOSTrainers.py / OnHarmonyTrainers.py / ToothFairy2Trainers.py):
  cp datasets/autopet/5_scripts_autopet/02_nnunet/AutoPETTrainers.py \
     .venv/lib/python3.*/site-packages/nnunetv2/training/nnUNetTrainer/

It imports the real implementations from the `autopet` package
(PROJECT_ROOT/datasets/autopet/5_scripts_autopet/), resolved via the NNUNET_PROJECT_ROOT
env var set by the 04_train/*.sh scripts. Method-base trainers (nnUNetTrainerFast,
AugLab's nnUNetTrainerDAExtGPU) live in src/ and the AugLab package.
"""
import os
import sys

_root = os.environ.get("NNUNET_PROJECT_ROOT", "")
if _root:
    _scripts = os.path.join(_root, "datasets", "autopet", "5_scripts_autopet")
    for _p in (_root, _scripts):
        if _p and _p not in sys.path:
            sys.path.insert(0, _p)

try:
    from autopet.trainers.baseline import nnUNetTrainerAutoPETBaseline              # noqa: F401
    from autopet.trainers.auglab_default import nnUNetTrainerAutoPETAugLabDefault    # noqa: F401
    from autopet.trainers.auglab_valsynth import nnUNetTrainerAutoPETAugLabValSynth  # noqa: F401
    from autopet.trainers.auglab_dualval import nnUNetTrainerAutoPETAugLabDualVal    # noqa: F401
except Exception as _e:  # pragma: no cover — discovery must not hard-crash nnU-Net import
    print(f"[AutoPETTrainers] import failed: {_e}")
