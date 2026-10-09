"""
nnUNetTrainerBraTS2024GliomaT2fAugLabDefault — AugLab GPU augmentation for BraTS 2024 T2f/FLAIR.

Thin T2f binding of nnUNetTrainerBraTS2024GliomaAugLabDefault. Identical augmentation
logic; only the trainer name differs so the results path correctly identifies T2f/FLAIR.
Used by: auglab_default, synthseg_EM, synthseg_noEM, srcsm, and the kmeans-ladder T2f
training scripts.
Uses Dataset053_BraTS2024GliomaT2f.

MRO: nnUNetTrainerBraTS2024GliomaT2fAugLabDefault
  → nnUNetTrainerBraTS2024GliomaAugLabDefault
  → nnUNetTrainerBraTS2024GliomaBase
  → nnUNetTrainerFast / nnUNetTrainerDAExtGPU / nnUNetTrainer
"""
from __future__ import annotations

from brats2024_glioma.trainers.auglab_default import nnUNetTrainerBraTS2024GliomaAugLabDefault


class nnUNetTrainerBraTS2024GliomaT2fAugLabDefault(nnUNetTrainerBraTS2024GliomaAugLabDefault):
    """AugLab default for T2f/FLAIR — same logic as T1n, different dataset + trainer name."""
    pass
