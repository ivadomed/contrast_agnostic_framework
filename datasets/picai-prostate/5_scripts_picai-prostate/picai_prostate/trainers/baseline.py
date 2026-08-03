"""
nnUNetTrainerPICAIProstateBaseline — picai-prostate real-data baseline (no synthesis).

Single input channel (T2W or ADC, per the training modality), standard nnUNet
augmentation only. The real-data reference against which the domain-randomization methods
(v26_6_2 + AugLab) and the SynthSeg-EM contender are compared for cross-contrast
(T2W→ADC/HBV, ADC→T2W/HBV) generalization.

Inherits split validation from nnUNetTrainerPICAIProstateBase and seed/epochs/WandB hooks
from nnUNetTrainerFast.
"""
from __future__ import annotations

from picai_prostate.trainers.base import nnUNetTrainerPICAIProstateBase


class nnUNetTrainerPICAIProstateBaseline(nnUNetTrainerPICAIProstateBase):
    """Standard-augmentation baseline (no synthesis) for picai-prostate."""
