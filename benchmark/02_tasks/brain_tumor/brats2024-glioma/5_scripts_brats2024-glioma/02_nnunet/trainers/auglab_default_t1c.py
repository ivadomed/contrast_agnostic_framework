"""
nnUNetTrainerBraTS2024GliomaT1cAugLabDefault — AugLab GPU augmentation for BraTS 2024 T1c.

Thin T1c binding of nnUNetTrainerBraTS2024GliomaAugLabDefault. Identical augmentation
logic; only the trainer name differs so the results path correctly identifies T1c.
Used by: auglab_default, synthseg_EM, synthseg_noEM, srcsm, and the kmeans-ladder T1c
training scripts.
Uses Dataset054_BraTS2024GliomaT1c.

MRO: nnUNetTrainerBraTS2024GliomaT1cAugLabDefault
  → nnUNetTrainerBraTS2024GliomaAugLabDefault
  → nnUNetTrainerBraTS2024GliomaBase
  → nnUNetTrainerFast / nnUNetTrainerDAExtGPU / nnUNetTrainer
"""
from __future__ import annotations

from brats2024_glioma.trainers.auglab_default import nnUNetTrainerBraTS2024GliomaAugLabDefault


class nnUNetTrainerBraTS2024GliomaT1cAugLabDefault(nnUNetTrainerBraTS2024GliomaAugLabDefault):
    """AugLab default for T1c — same logic as T1n, different dataset + trainer name."""
    pass
