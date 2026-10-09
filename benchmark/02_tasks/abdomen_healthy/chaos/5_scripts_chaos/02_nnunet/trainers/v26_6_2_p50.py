"""
nnUNetTrainerCHAOSV26_6_2_p50 — V26_6_2 for CHAOS with 50% training synth probability.

Same as nnUNetTrainerCHAOSV26_6_2 in every respect except train_synth_prob = 0.5.
Intended for the exp1 ablation to compare 50/100 vs the default 90/100.
"""
from __future__ import annotations

from chaos.trainers.v26_6_2 import nnUNetTrainerCHAOSV26_6_2


class nnUNetTrainerCHAOSV26_6_2_p50(nnUNetTrainerCHAOSV26_6_2):
    """V26_6_2 for CHAOS with train_synth_prob=0.5, val_synth_prob=1.0 (exp1 ablation)."""

    train_synth_prob: float = 0.5
