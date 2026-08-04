"""
nnUNetTrainerAtlasLiverHCCBaseline — atlas-liver-hcc real-data baseline (no synthesis).

Single T1w CE-MRI input channel, standard nnUNet augmentation only. The real-data
reference against which the domain-randomization methods (v26_6_2 + AugLab) and the
SynthSeg-EM contender are compared. Single-modality dataset — no cross-contrast
generalization axis, unlike open-ms/chaos/brats (see 00_utils/env.sh): the held-out
test-patient split IS the generalization test here.

Inherits split validation from nnUNetTrainerAtlasLiverHCCBase and seed/epochs/WandB
hooks from nnUNetTrainerFast.
"""
from __future__ import annotations

from atlas_liver_hcc.trainers.base import nnUNetTrainerAtlasLiverHCCBase


class nnUNetTrainerAtlasLiverHCCBaseline(nnUNetTrainerAtlasLiverHCCBase):
    """Standard-augmentation baseline (no synthesis) for atlas-liver-hcc."""
