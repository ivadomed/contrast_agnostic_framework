# autopet dataset-specific Python package (AutoPET FDG+PSMA whole-body PET/CT tumor-lesion
# segmentation, training set as of the 2026-09-13 HECKTOR-drop pivot).
# env.sh adds 5_scripts_autopet/ to PYTHONPATH so this imports as `autopet`:
#   from autopet.trainers.baseline import nnUNetTrainerAutoPETBaseline
# nnU-Net discovers the concrete trainer classes via the registration shim
# 02_nnunet/AutoPETTrainers.py (copied into the installed nnunetv2 package).
#
# Adapted from datasets/toothfairy2/5_scripts_toothfairy2/toothfairy2/ (closest structural
# analog: two-training-modality lesion/tumor segmentation with the full 6-method suite +
# causal-ablation ladder pattern) — N_EXPECTED_FOLDS=3 (this project's permanent fold
# policy), see trainers/base.py.
__all__: list[str] = []
from autopet.trainers.baseline import nnUNetTrainerAutoPETBaseline
from autopet.trainers.auglab_default import nnUNetTrainerAutoPETAugLabDefault
from autopet.trainers.auglab_valsynth import nnUNetTrainerAutoPETAugLabValSynth
from autopet.trainers.auglab_dualval import nnUNetTrainerAutoPETAugLabDualVal

__all__ = [
    "nnUNetTrainerAutoPETBaseline",
    "nnUNetTrainerAutoPETAugLabDefault",
    "nnUNetTrainerAutoPETAugLabValSynth",
    "nnUNetTrainerAutoPETAugLabDualVal",
]
