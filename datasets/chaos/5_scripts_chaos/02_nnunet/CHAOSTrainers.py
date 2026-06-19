# Backup — NOT imported by anything in this repo. nnU-Net's recursive_find_python_class
# only searches inside the installed nnunetv2 package's training/nnUNetTrainer/ dir, so
# this registration shim must be copied there after every fresh venv/nnunetv2 install:
#   cp datasets/chaos/5_scripts_chaos/02_nnunet/CHAOSTrainers.py \
#      .venv/lib/python3.*/site-packages/nnunetv2/training/nnUNetTrainer/
# Found to exist only inside .venv during the set_slot -> Slurm migration (2026-06-19) —
# there was no backup anywhere, so a venv rebuild would have silently broken discovery
# of every nnUNetTrainerCHAOS* class (training would fail with "trainer not found" for
# this dataset).
"""
Registration file — makes CHAOS custom trainers discoverable by nnUNet.

nnUNet searches this directory for trainer classes via recursive_find_python_class().
This file imports the real implementations from the `chaos` package, which lives at
PROJECT_ROOT/datasets/chaos/5_scripts_chaos/chaos/ (added to the path here and via
env.sh's PYTHONPATH).

Project root is resolved from the NNUNET_PROJECT_ROOT environment variable set by the
training/prediction scripts (04_train/*.sh, 05_predict/*.sh).
"""
import os
import sys

_root = os.environ.get("NNUNET_PROJECT_ROOT", "")
if _root:
    _scripts = os.path.join(_root, "datasets", "chaos", "5_scripts_chaos")
    for _p in (_root, _scripts):
        if _p and _p not in sys.path:
            sys.path.insert(0, _p)

try:
    from chaos.trainers.baseline import nnUNetTrainerCHAOSBaseline  # noqa: F401
    from chaos.trainers.v26_6_2 import nnUNetTrainerCHAOSV26_6_2  # noqa: F401
    from chaos.trainers.v26_6_2_p50 import nnUNetTrainerCHAOSV26_6_2_p50  # noqa: F401
    from chaos.trainers.v26_6_2_train050_val000 import nnUNetTrainerCHAOSV26_6_2_train050_val000  # noqa: F401
    from chaos.trainers.v26_6_2_train025_val000 import nnUNetTrainerCHAOSV26_6_2_train025_val000  # noqa: F401
    from chaos.trainers.v26_6_2_train025_val100 import nnUNetTrainerCHAOSV26_6_2_train025_val100  # noqa: F401
    from chaos.trainers.auglab_default import nnUNetTrainerCHAOSAugLabDefault  # noqa: F401
    from chaos.trainers.auglab_v26_6_2 import nnUNetTrainerCHAOSAugLabV26_6_2  # noqa: F401
    from chaos.trainers.auglab_valsynth import nnUNetTrainerCHAOSAugLabValSynth  # noqa: F401
except ImportError as e:
    import warnings
    warnings.warn(
        f"[CHAOSTrainers] Import failed: {e}. "
        "Make sure NNUNET_PROJECT_ROOT is set to the mri_synthesis_project root "
        "and that datasets/chaos/5_scripts_chaos is importable."
    )
