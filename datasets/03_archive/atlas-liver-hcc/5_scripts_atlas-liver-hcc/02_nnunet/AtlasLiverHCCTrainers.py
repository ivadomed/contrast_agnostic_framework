"""
Registration shim — makes atlas-liver-hcc custom trainers discoverable by nnU-Net.

nnU-Net's recursive_find_python_class() only searches inside the installed nnunetv2
package's training/nnUNetTrainer/ dir, so this file must be copied there after every
fresh venv/nnunetv2 install:
  cp datasets/atlas-liver-hcc/5_scripts_atlas-liver-hcc/02_nnunet/AtlasLiverHCCTrainers.py \
     .venv/lib/python3.*/site-packages/nnunetv2/training/nnUNetTrainer/

It imports the real implementations from the `atlas_liver_hcc` package
(PROJECT_ROOT/datasets/atlas-liver-hcc/5_scripts_atlas-liver-hcc/), resolved via the
NNUNET_PROJECT_ROOT env var set by the 04_train/*.sh scripts. Method-base trainers
(nnUNetTrainerFast, AugLab's nnUNetTrainerDAExtGPU) live in src/ and the AugLab package.
"""
import os
import sys

_root = os.environ.get("NNUNET_PROJECT_ROOT", "")
if _root:
    _scripts = os.path.join(_root, "datasets", "atlas-liver-hcc", "5_scripts_atlas-liver-hcc")
    for _p in (_root, _scripts):
        if _p and _p not in sys.path:
            sys.path.insert(0, _p)

try:
    from atlas_liver_hcc.trainers.baseline import nnUNetTrainerAtlasLiverHCCBaseline              # noqa: F401
    from atlas_liver_hcc.trainers.auglab_default import nnUNetTrainerAtlasLiverHCCAugLabDefault    # noqa: F401
    from atlas_liver_hcc.trainers.auglab_valsynth import nnUNetTrainerAtlasLiverHCCAugLabValSynth  # noqa: F401
    from atlas_liver_hcc.trainers.auglab_dualval import nnUNetTrainerAtlasLiverHCCAugLabDualVal    # noqa: F401  # opt-in only, see module docstring
except Exception as _e:  # pragma: no cover — discovery must not hard-crash nnU-Net import
    print(f"[AtlasLiverHCCTrainers] import failed: {_e}")
