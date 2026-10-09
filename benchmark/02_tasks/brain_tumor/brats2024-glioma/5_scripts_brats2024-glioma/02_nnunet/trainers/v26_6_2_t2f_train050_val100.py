"""
nnUNetTrainerBraTS2024GliomaT2fV26_6_2_train050_val100

V26_6_2 T2f/FLAIR with train_synth_prob=0.5, val_synth_prob=1.0.
Parallel to nnUNetTrainerBraTS2024GliomaV26_6_2_train050_val100 (T1n).
Uses Dataset053_BraTS2024GliomaT2f.
"""
from __future__ import annotations

from brats2024_glioma.trainers.v26_6_2_t2f import nnUNetTrainerBraTS2024GliomaT2fV26_6_2


class nnUNetTrainerBraTS2024GliomaT2fV26_6_2_train050_val100(nnUNetTrainerBraTS2024GliomaT2fV26_6_2):
    train_synth_prob: float = 0.5
    val_synth_prob: float = 1.0
