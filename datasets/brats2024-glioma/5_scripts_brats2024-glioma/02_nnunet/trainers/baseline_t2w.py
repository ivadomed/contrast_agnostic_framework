"""
nnUNetTrainerBraTS2024GliomaT2wBaseline — T2w-only BraTS 2024 baseline.

Single input channel (T2 weighted). Parallel to nnUNetTrainerBraTS2024GliomaT1nBaseline
but uses Dataset052_BraTS2024GliomaT2w. Trainer name is distinct so the nnUNet results
directory path clearly identifies T2w training.

MRO: nnUNetTrainerBraTS2024GliomaT2wBaseline
  → nnUNetTrainerBraTS2024GliomaT1nBaseline
  → nnUNetTrainerBraTS2024GliomaBase  (anti-contamination do_split)
  → nnUNetTrainerFast                  (seed, epochs, WandB hooks)
  → nnUNetTrainer
"""
from __future__ import annotations

from brats2024_glioma.trainers.baseline_t1n import nnUNetTrainerBraTS2024GliomaT1nBaseline


class nnUNetTrainerBraTS2024GliomaT2wBaseline(nnUNetTrainerBraTS2024GliomaT1nBaseline):
    """T2w baseline — same logic as T1n baseline, different dataset + trainer name."""
    pass
