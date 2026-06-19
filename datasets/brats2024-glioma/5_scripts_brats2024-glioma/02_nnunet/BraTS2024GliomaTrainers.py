# Backup — NOT imported by anything in this repo. nnU-Net's recursive_find_python_class
# only searches inside the installed nnunetv2 package's training/nnUNetTrainer/ dir, so
# this registration shim must be copied there after every fresh venv/nnunetv2 install:
#   cp datasets/brats2024-glioma/5_scripts_brats2024-glioma/02_nnunet/BraTS2024GliomaTrainers.py \
#      .venv/lib/python3.*/site-packages/nnunetv2/training/nnUNetTrainer/
# Found to exist only inside .venv during the set_slot -> Slurm migration (2026-06-19) —
# there was no backup anywhere, so a venv rebuild would have silently broken discovery
# of every nnUNetTrainerBraTS2024Glioma* class (training would fail with "trainer not
# found" for this dataset).
"""
Registration file — makes BraTS 2024 Glioma custom trainers discoverable by nnUNet.

nnUNet searches this directory for trainer classes via recursive_find_python_class().
This file imports the real implementations from the brats2024_glioma package, which
lives at PROJECT_ROOT/datasets/brats2024-glioma/5_scripts_brats2024-glioma/brats2024_glioma/
(added to the path here and via env.sh's PYTHONPATH).

Project root is resolved from the NNUNET_PROJECT_ROOT environment variable set by
the training scripts (04_train/*.sh).
"""
import os
import sys

_root = os.environ.get("NNUNET_PROJECT_ROOT", "")
if _root:
    _scripts = os.path.join(_root, "datasets", "brats2024-glioma", "5_scripts_brats2024-glioma")
    for _p in (_root, _scripts):
        if _p and _p not in sys.path:
            sys.path.insert(0, _p)

try:
    from brats2024_glioma.trainers.baseline_t1n import nnUNetTrainerBraTS2024GliomaT1nBaseline  # noqa: F401
    from brats2024_glioma.trainers.v26_6 import nnUNetTrainerBraTS2024GliomaV26_6  # noqa: F401
    from brats2024_glioma.trainers.v26_6_2 import nnUNetTrainerBraTS2024GliomaV26_6_2  # noqa: F401
    from brats2024_glioma.trainers.auglab_default import nnUNetTrainerBraTS2024GliomaAugLabDefault  # noqa: F401
    from brats2024_glioma.trainers.auglab_default_valaug import nnUNetTrainerBraTS2024GliomaAugLabDefaultValAug  # noqa: F401
    from brats2024_glioma.trainers.auglab_valsynth import nnUNetTrainerBraTS2024GliomaAugLabValSynth  # noqa: F401
    from brats2024_glioma.trainers.v26_6_2_train050_val100 import nnUNetTrainerBraTS2024GliomaV26_6_2_train050_val100  # noqa: F401
except ImportError as e:
    import warnings
    warnings.warn(
        f"[BraTS2024GliomaTrainers] Import failed: {e}. "
        "Make sure NNUNET_PROJECT_ROOT is set to the mri_synthesis_project root "
        "and that datasets/brats2024-glioma/5_scripts_brats2024-glioma is importable."
    )
