"""
nnUNetTrainerBraTS2024GliomaT1cV26_6_2_train050_val100

V26_6_2 T1c with train_synth_prob=0.5, val_synth_prob=1.0.
Parallel to nnUNetTrainerBraTS2024GliomaV26_6_2_train050_val100 (T1n).
Uses Dataset054_BraTS2024GliomaT1c.
"""
from __future__ import annotations

from brats2024_glioma.trainers.v26_6_2_t1c import nnUNetTrainerBraTS2024GliomaT1cV26_6_2


class nnUNetTrainerBraTS2024GliomaT1cV26_6_2_train050_val100(nnUNetTrainerBraTS2024GliomaT1cV26_6_2):
    train_synth_prob: float = 0.5
    val_synth_prob: float = 1.0
