"""
nnUNetTrainerCHAOSV26_6_2_train025_val000 — V26_6_2 for CHAOS with 25% train synth,
0% val synth.

Same as nnUNetTrainerCHAOSV26_6_2 except train_synth_prob=0.25, val_synth_prob=0.0:
validation always runs on the real (z-scored) volume, never synthesised.
"""
from __future__ import annotations

from chaos.trainers.v26_6_2 import nnUNetTrainerCHAOSV26_6_2


class nnUNetTrainerCHAOSV26_6_2_train025_val000(nnUNetTrainerCHAOSV26_6_2):
    """V26_6_2 for CHAOS: train_synth_prob=0.25, val_synth_prob=0.0."""

    train_synth_prob: float = 0.25
    val_synth_prob: float = 0.0
