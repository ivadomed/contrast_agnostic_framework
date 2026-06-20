"""
nnUNetTrainerBraTS2024GliomaT2wAugLabDefault — AugLab GPU augmentation for BraTS 2024 T2w.

Thin T2w binding of nnUNetTrainerBraTS2024GliomaAugLabDefault. Identical augmentation
logic; only the trainer name differs so the results path correctly identifies T2w.
Used by: auglab_default, synthseg_EM, synthseg_noEM T2w training scripts.
Uses Dataset052_BraTS2024GliomaT2w.

MRO: nnUNetTrainerBraTS2024GliomaT2wAugLabDefault
  → nnUNetTrainerBraTS2024GliomaAugLabDefault
  → nnUNetTrainerBraTS2024GliomaBase
  → nnUNetTrainerFast / nnUNetTrainerDAExtGPU / nnUNetTrainer
"""
from __future__ import annotations

from brats2024_glioma.trainers.auglab_default import nnUNetTrainerBraTS2024GliomaAugLabDefault


class nnUNetTrainerBraTS2024GliomaT2wAugLabDefault(nnUNetTrainerBraTS2024GliomaAugLabDefault):
    """AugLab default for T2w — same logic as T1n, different dataset + trainer name."""
    pass
