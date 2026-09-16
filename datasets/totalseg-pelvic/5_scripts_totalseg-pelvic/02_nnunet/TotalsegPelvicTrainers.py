"""
Registration shim — makes totalseg-pelvic custom trainers discoverable by nnU-Net.

nnU-Net's recursive_find_python_class() only searches inside the installed nnunetv2
package's training/nnUNetTrainer/ dir, so this file must be copied there after every
fresh venv/nnunetv2 install, ON EVERY CLUSTER (Vulcan and TamIA separately — see
CLAUDE.md's "registration shims" section; same mechanism as AutoPETTrainers.py /
CHAOSTrainers.py / OnHarmonyTrainers.py / ToothFairy2Trainers.py):
  cp datasets/totalseg-pelvic/5_scripts_totalseg-pelvic/02_nnunet/TotalsegPelvicTrainers.py \
     .venv/lib/python3.*/site-packages/nnunetv2/training/nnUNetTrainer/

It imports the real implementations from the `totalseg_pelvic` package
(PROJECT_ROOT/datasets/totalseg-pelvic/5_scripts_totalseg-pelvic/), resolved via the
NNUNET_PROJECT_ROOT env var set by the 04_train/*.sh scripts.
"""
import os
import sys

_root = os.environ.get("NNUNET_PROJECT_ROOT", "")
if _root:
    _scripts = os.path.join(_root, "datasets", "totalseg-pelvic", "5_scripts_totalseg-pelvic")
    for _p in (_root, _scripts):
        if _p and _p not in sys.path:
            sys.path.insert(0, _p)

try:
    from totalseg_pelvic.trainers.baseline import nnUNetTrainerTotalsegPelvicBaseline              # noqa: F401
    from totalseg_pelvic.trainers.auglab_default import nnUNetTrainerTotalsegPelvicAugLabDefault    # noqa: F401
    from totalseg_pelvic.trainers.auglab_valsynth import nnUNetTrainerTotalsegPelvicAugLabValSynth  # noqa: F401
    from totalseg_pelvic.trainers.auglab_dualval import nnUNetTrainerTotalsegPelvicAugLabDualVal    # noqa: F401
except Exception as _e:  # pragma: no cover — discovery must not hard-crash nnU-Net import
    print(f"[TotalsegPelvicTrainers] import failed: {_e}")
