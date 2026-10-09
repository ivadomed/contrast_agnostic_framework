"""
Registration shim — makes isles2022 custom trainers discoverable by nnU-Net.

nnU-Net's recursive_find_python_class() only searches inside the installed nnunetv2
package's training/nnUNetTrainer/ dir, so this file must be copied there after every
fresh venv/nnunetv2 install, ON EVERY CLUSTER (Vulcan and TamIA separately):
  cp benchmark/02_tasks/brain_stroke/isles2022/5_scripts_isles2022/02_nnunet/ISLES2022Trainers.py \
     .venv/lib/python3.*/site-packages/nnunetv2/training/nnUNetTrainer/
(The venv-setup script auto-discovers *Trainers.py under benchmark/02_tasks; see the project notes.)

It imports the real implementations from the `isles2022` package
(PROJECT_ROOT/benchmark/02_tasks/brain_stroke/isles2022/5_scripts_isles2022/), resolved via the
NNUNET_PROJECT_ROOT env var set by the 04_train/*.sh scripts.
"""
import os
import sys

_root = os.environ.get("NNUNET_PROJECT_ROOT", "")
if _root:
    _scripts = os.path.join(_root, "benchmark", "02_tasks", "brain_stroke", "isles2022", "5_scripts_isles2022")
    for _p in (_root, _scripts):
        if _p and _p not in sys.path:
            sys.path.insert(0, _p)

try:
    from isles2022.trainers.baseline import nnUNetTrainerISLES2022Baseline              # noqa: F401
    from isles2022.trainers.auglab_default import nnUNetTrainerISLES2022AugLabDefault    # noqa: F401
    from isles2022.trainers.auglab_valsynth import nnUNetTrainerISLES2022AugLabValSynth  # noqa: F401
    from isles2022.trainers.auglab_dualval import nnUNetTrainerISLES2022AugLabDualVal    # noqa: F401
except Exception as _e:  # pragma: no cover — discovery must not hard-crash nnU-Net import
    print(f"[ISLES2022Trainers] import failed: {_e}")
