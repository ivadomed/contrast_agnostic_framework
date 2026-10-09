"""
nnUNetTrainerBraTS2024GliomaV26_6_2_train050_val100

V26_6_2 with train_synth_prob=0.5 (instead of the default 0.9).
Val_synth_prob=1.0 (same as the base).
"""
from __future__ import annotations

from brats2024_glioma.trainers.v26_6_2 import nnUNetTrainerBraTS2024GliomaV26_6_2


class nnUNetTrainerBraTS2024GliomaV26_6_2_train050_val100(nnUNetTrainerBraTS2024GliomaV26_6_2):
    train_synth_prob: float = 0.5
    val_synth_prob: float = 1.0
