"""
nnUNetTrainerCHAOSV26_6_2_train025_val100 — V26_6_2 for CHAOS with 25% train synth,
100% val synth.

Same as nnUNetTrainerCHAOSV26_6_2 (default: 90%/100%) except train_synth_prob=0.25;
val_synth_prob stays 1.0 so validation always runs on a synthesised volume.
"""
from __future__ import annotations

from chaos.trainers.v26_6_2 import nnUNetTrainerCHAOSV26_6_2


class nnUNetTrainerCHAOSV26_6_2_train025_val100(nnUNetTrainerCHAOSV26_6_2):
    """V26_6_2 for CHAOS: train_synth_prob=0.25, val_synth_prob=1.0."""

    train_synth_prob: float = 0.25
    val_synth_prob: float = 1.0
