"""
nnUNetTrainerOnHarmonyV26_6_2 — V26_6_2 trainer for the ON-Harmony dataset.

Adds ON-Harmony-specific data loading to nnUNetTrainerV26_6_2.
Inherits get_dataloaders from nnUNetTrainerOnHarmonyV26_6 and the
label-guided synthesis train_step from nnUNetTrainerV26_6_2.
"""
from __future__ import annotations

from on_harmony.trainers.v26_6 import nnUNetTrainerOnHarmonyV26_6
from src.nnunet.trainers.v26_6_2_base import nnUNetTrainerV26_6_2


class nnUNetTrainerOnHarmonyV26_6_2(nnUNetTrainerOnHarmonyV26_6, nnUNetTrainerV26_6_2):
    """
    V26_6_2 trainer for ON-Harmony.

    MRO: nnUNetTrainerOnHarmonyV26_6_2
      → nnUNetTrainerOnHarmonyV26_6  (get_dataloaders — OnHarmonyVolumeLoader)
      → nnUNetTrainerOnHarmonyBase   (do_split — anti-contamination guard)
      → nnUNetTrainerV26_6_2         (train_step — label-guided synthesis)
      → nnUNetTrainerV26_6           (validation_step, transforms, WandB logging)
      → nnUNetTrainerFast            (seed, epochs)
      → nnUNetTrainer
    """
