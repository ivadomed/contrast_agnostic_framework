"""
nnUNetTrainerOnHarmonyV26_6_2_train050_val100 — V26_6_2 for ON-Harmony, 50% train synth.

Identical backend to nnUNetTrainerOnHarmonyV26_6_2 (AugLab contrast transform, see that
module). The 50%/100% train/val synth split is set by the config JSONs the 04_train
wrapper exports (train050 + VALsynthonly), NOT by class attributes — this class exists
only as a distinct trainer name so its runs get their own model dir.
"""
from __future__ import annotations

from on_harmony.trainers.v26_6_2 import nnUNetTrainerOnHarmonyV26_6_2


class nnUNetTrainerOnHarmonyV26_6_2_train050_val100(nnUNetTrainerOnHarmonyV26_6_2):
    """V26_6_2 for ON-Harmony (50% train / 100% val synth — set via config JSON)."""
