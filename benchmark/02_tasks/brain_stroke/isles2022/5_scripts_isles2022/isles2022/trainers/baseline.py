"""
nnUNetTrainerISLES2022Baseline — isles2022 real-data baseline (no synthesis).

Single input channel (dwi OR flair, depending on which per-modality Dataset it's
trained against — DATASET_ID selects this in the 04_0X wrapper, not this class),
standard nnUNet augmentation only. The real-data reference against which the
domain-randomization methods (v26_6_2 + AugLab) and the SynthSeg-EM contender are
compared. Two training modalities (like ambl/chaos/brats2024-glioma) — see
00_utils/env.sh / env_flair.sh: each modality is its own nnU-Net Dataset
(140=dwi, 141=flair), held-out test patients scored on dwi, adc AND flair for the
cross-contrast generalization axis.

Inherits split validation from nnUNetTrainerISLES2022Base and seed/epochs/WandB hooks
from nnUNetTrainerFast.
"""
from __future__ import annotations

from isles2022.trainers.base import nnUNetTrainerISLES2022Base


class nnUNetTrainerISLES2022Baseline(nnUNetTrainerISLES2022Base):
    """Standard-augmentation baseline (no synthesis) for isles2022."""
