# Backup — NOT imported by anything in this repo. nnU-Net's recursive_find_python_class
# only searches inside the installed nnunetv2 package's training/nnUNetTrainer/ dir, so
# this registration shim must be copied there after every fresh venv/nnunetv2 install:
#   cp datasets/on-harmony/5_scripts_on-harmony/02_nnunet/OnHarmonyTrainers.py \
#      .venv/lib/python3.*/site-packages/nnunetv2/training/nnUNetTrainer/
# Found to exist only inside .venv during the set_slot -> Slurm migration (2026-06-19) —
# there was no backup anywhere, so a venv rebuild would have silently broken discovery
# of every nnUNetTrainerOnHarmony* class (training would fail with "trainer not found"
# for this dataset).
"""
Registration file — makes ON-Harmony custom trainers discoverable by nnUNet.

nnUNet searches this directory for trainer classes via recursive_find_python_class().
This file imports the real implementations from the on_harmony package, which lives
at PROJECT_ROOT/datasets/on-harmony/5_scripts_on-harmony/on_harmony/ (added to the
path here and via env.sh's PYTHONPATH).  Method-base trainers live in
PROJECT_ROOT/src/nnunet/trainers/.

Project root is resolved from the NNUNET_PROJECT_ROOT environment variable set by
the training scripts (04_train/*.sh).
"""
import os
import sys

_root = os.environ.get("NNUNET_PROJECT_ROOT", "")
if _root:
    _scripts = os.path.join(_root, "datasets", "on-harmony", "5_scripts_on-harmony")
    for _p in (_root, _scripts):
        if _p and _p not in sys.path:
            sys.path.insert(0, _p)

try:
    from on_harmony.trainers.baseline import nnUNetTrainerOnHarmonyBaseline                                    # noqa: F401
    from on_harmony.trainers.v26_6 import nnUNetTrainerOnHarmonyV26_6                                         # noqa: F401
    from on_harmony.trainers.v26_6_2 import nnUNetTrainerOnHarmonyV26_6_2                                     # noqa: F401
    from on_harmony.trainers.v26_6_2_train050_val100 import nnUNetTrainerOnHarmonyV26_6_2_train050_val100      # noqa: F401
    from on_harmony.trainers.synthseg_a import nnUNetTrainerOnHarmonySynthSegA                                 # noqa: F401
    from on_harmony.trainers.synthseg_b import nnUNetTrainerOnHarmonySynthSegB                                 # noqa: F401
    from on_harmony.trainers.auglab_default import nnUNetTrainerOnHarmonyAugLabDefault                         # noqa: F401
    from on_harmony.trainers.auglab_valsynth import nnUNetTrainerOnHarmonyAugLabValSynth                       # noqa: F401
except ImportError as e:
    import warnings
    warnings.warn(
        f"[OnHarmonyTrainers] Import failed: {e}. "
        "Make sure NNUNET_PROJECT_ROOT is set to the mri_synthesis_project root "
        "and that datasets/on-harmony/5_scripts_on-harmony is importable."
    )
