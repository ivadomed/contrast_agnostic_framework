"""
nnUNetTrainerBraTS2024GliomaT2fBaseline — T2f/FLAIR-only BraTS 2024 baseline.

Single input channel (FLAIR). Parallel to nnUNetTrainerBraTS2024GliomaT1nBaseline
but uses Dataset053_BraTS2024GliomaT2f. Trainer name is distinct so the nnUNet results
directory path clearly identifies T2f/FLAIR training.

MRO: nnUNetTrainerBraTS2024GliomaT2fBaseline
  → nnUNetTrainerBraTS2024GliomaT1nBaseline
  → nnUNetTrainerBraTS2024GliomaBase  (anti-contamination do_split)
  → nnUNetTrainerFast                  (seed, epochs, WandB hooks)
  → nnUNetTrainer
"""
from __future__ import annotations

from brats2024_glioma.trainers.baseline_t1n import nnUNetTrainerBraTS2024GliomaT1nBaseline


class nnUNetTrainerBraTS2024GliomaT2fBaseline(nnUNetTrainerBraTS2024GliomaT1nBaseline):
    """T2f/FLAIR baseline — same logic as T1n baseline, different dataset + trainer name."""
    pass
