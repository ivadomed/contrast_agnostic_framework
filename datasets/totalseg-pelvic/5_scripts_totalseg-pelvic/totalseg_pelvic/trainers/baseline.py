"""
nnUNetTrainerTotalsegPelvicBaseline — totalseg-pelvic real-data baseline (no synthesis,
standard nnU-Net augmentation, no FOV-widening override — unlike autopet, no measured
FOV/train-eval mismatch has been found for this dataset; revisit only if one is measured).

Single input channel (CT OR MRI, depending on which per-modality Dataset it's trained
against — DATASET_ID selects this in the 04_0X wrapper, not this class). The real-data
reference against which the domain-randomization methods (v26_6_2 + AugLab) and the
SynthSeg-EM contender are compared. Two training modalities (like chaos/autopet/ispy2) —
see 00_utils/env.sh / env_mri.sh: each modality is its own nnU-Net Dataset, held-out test
cases scored on BOTH modalities for the cross-contrast axis.

Inherits split validation from nnUNetTrainerTotalsegPelvicBase and seed/epochs/WandB hooks
from nnUNetTrainerFast.
"""
from __future__ import annotations

from totalseg_pelvic.trainers.base import nnUNetTrainerTotalsegPelvicBase


class nnUNetTrainerTotalsegPelvicBaseline(nnUNetTrainerTotalsegPelvicBase):
    """Standard-augmentation baseline (no synthesis) for totalseg-pelvic."""
