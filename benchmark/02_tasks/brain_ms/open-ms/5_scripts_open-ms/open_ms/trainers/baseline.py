"""
nnUNetTrainerOpenMSBaseline — open-ms real-data baseline (no synthesis).

Single FLAIR input channel, standard nnUNet augmentation only. The real-data
reference against which the domain-randomization methods (v26_6_2 + AugLab) and the
SynthSeg-EM contender are compared for cross-contrast (FLAIR→T2W/T1W) generalization.

Inherits split validation from nnUNetTrainerOpenMSBase and seed/epochs/WandB hooks
from nnUNetTrainerFast.
"""
from __future__ import annotations

from open_ms.trainers.base import nnUNetTrainerOpenMSBase


class nnUNetTrainerOpenMSBaseline(nnUNetTrainerOpenMSBase):
    """Standard-augmentation baseline (no synthesis) for open-ms."""
