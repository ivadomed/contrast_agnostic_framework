"""
nnUNetTrainerToothFairy2Baseline — toothfairy2 real-data baseline (no synthesis).

Single input channel (t1wce OR t2w, depending on which per-modality Dataset it's
trained against — DATASET_ID selects this in the 04_0X wrapper, not this class),
standard nnUNet augmentation only. The real-data reference against which the
domain-randomization methods (v26_6_2 + AugLab) and the SynthSeg-EM contender are
compared. Two training modalities (like ambl/chaos/brats2024-glioma) — see
00_utils/env.sh / env_t2w.sh: each modality is its own nnU-Net Dataset
(100=t1wce, 101=t2w), held-out test patients scored on BOTH modalities for the
cross-contrast generalization axis.

Inherits split validation from nnUNetTrainerToothFairy2Base and seed/epochs/WandB hooks
from nnUNetTrainerFast.
"""
from __future__ import annotations

from toothfairy2.trainers.base import nnUNetTrainerToothFairy2Base


class nnUNetTrainerToothFairy2Baseline(nnUNetTrainerToothFairy2Base):
    """Standard-augmentation baseline (no synthesis) for toothfairy2."""
