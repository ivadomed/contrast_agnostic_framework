"""
nnUNetTrainerBraTS2024GliomaT2wV26_6_2 — V26_6_2 synthesis for BraTS 2024 T2w.

Thin T2w binding of V26_6_2. Identical synthesis logic to
nnUNetTrainerBraTS2024GliomaV26_6_2 (Dataset051, T1n); only the trainer name
differs so the nnUNet results path correctly identifies T2w training.
Uses Dataset052_BraTS2024GliomaT2w.

MRO: nnUNetTrainerBraTS2024GliomaT2wV26_6_2
  → nnUNetTrainerBraTS2024GliomaV26_6_2
  → nnUNetTrainerBraTS2024GliomaBase  (anti-contamination do_split)
  → nnUNetTrainerV26_6_2
  → nnUNetTrainerFast / nnUNetTrainer
"""
from __future__ import annotations

from brats2024_glioma.trainers.v26_6_2 import nnUNetTrainerBraTS2024GliomaV26_6_2


class nnUNetTrainerBraTS2024GliomaT2wV26_6_2(nnUNetTrainerBraTS2024GliomaV26_6_2):
    """V26_6_2 for T2w — same logic as T1n V26_6_2, different dataset + trainer name."""
    pass
