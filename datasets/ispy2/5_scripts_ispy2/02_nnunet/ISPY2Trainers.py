"""
Registration shim — makes ispy2 custom trainers discoverable by nnU-Net.

nnU-Net's recursive_find_python_class() only searches inside the installed nnunetv2
package's training/nnUNetTrainer/ dir, so this file must be copied there after every
fresh venv/nnunetv2 install, ON EVERY CLUSTER (Vulcan and TamIA separately — a real
gap that bit the ambl run: training there silently failed "trainer not found" until
this was caught and fixed):
  cp datasets/ispy2/5_scripts_ispy2/02_nnunet/ISPY2Trainers.py \
     .venv/lib/python3.*/site-packages/nnunetv2/training/nnUNetTrainer/

(See CLAUDE.md's "registration shims" section — same mechanism as
BraTS2024GliomaTrainers.py / CHAOSTrainers.py / OnHarmonyTrainers.py /
AMBLTrainers.py.)

It imports the real implementations from the `ispy2` package
(PROJECT_ROOT/datasets/ispy2/5_scripts_ispy2/), resolved via the NNUNET_PROJECT_ROOT
env var set by the 04_train/*.sh scripts. Method-base trainers (nnUNetTrainerFast,
AugLab's nnUNetTrainerDAExtGPU) live in src/ and the AugLab package.
"""
import os
import sys

_root = os.environ.get("NNUNET_PROJECT_ROOT", "")
if _root:
    _scripts = os.path.join(_root, "datasets", "ispy2", "5_scripts_ispy2")
    for _p in (_root, _scripts):
        if _p and _p not in sys.path:
            sys.path.insert(0, _p)

try:
    from ispy2.trainers.baseline import nnUNetTrainerISPY2Baseline              # noqa: F401
    from ispy2.trainers.auglab_default import nnUNetTrainerISPY2AugLabDefault    # noqa: F401
    from ispy2.trainers.auglab_valsynth import nnUNetTrainerISPY2AugLabValSynth  # noqa: F401
    from ispy2.trainers.auglab_dualval import nnUNetTrainerISPY2AugLabDualVal    # noqa: F401
except Exception as _e:  # pragma: no cover — discovery must not hard-crash nnU-Net import
    print(f"[ISPY2Trainers] import failed: {_e}")
