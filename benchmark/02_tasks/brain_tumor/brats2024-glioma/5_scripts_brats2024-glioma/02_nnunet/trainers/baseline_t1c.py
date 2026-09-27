"""
nnUNetTrainerBraTS2024GliomaT1cBaseline — T1c-only BraTS 2024 baseline.

Single input channel (T1c). Parallel to nnUNetTrainerBraTS2024GliomaT1nBaseline
but uses Dataset054_BraTS2024GliomaT1c. Trainer name is distinct so the nnUNet results
directory path clearly identifies T1c training.

MRO: nnUNetTrainerBraTS2024GliomaT1cBaseline
  → nnUNetTrainerBraTS2024GliomaT1nBaseline
  → nnUNetTrainerBraTS2024GliomaBase  (anti-contamination do_split)
  → nnUNetTrainerFast                  (seed, epochs, WandB hooks)
  → nnUNetTrainer
"""
from __future__ import annotations

from brats2024_glioma.trainers.baseline_t1n import nnUNetTrainerBraTS2024GliomaT1nBaseline


class nnUNetTrainerBraTS2024GliomaT1cBaseline(nnUNetTrainerBraTS2024GliomaT1nBaseline):
    """T1c baseline — same logic as T1n baseline, different dataset + trainer name."""
    pass
