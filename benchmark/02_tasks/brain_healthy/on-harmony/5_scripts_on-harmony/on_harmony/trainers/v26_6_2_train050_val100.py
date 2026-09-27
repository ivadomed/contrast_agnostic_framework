"""
nnUNetTrainerOnHarmonyV26_6_2_train050_val100 — V26_6_2 for ON-Harmony, 50% train synth.

Identical backend to nnUNetTrainerOnHarmonyV26_6_2 (AugLab contrast transform, see that
module). The 50%/100% train/val synth split is set by the config JSONs the 04_train
wrapper exports (train050 + VALsynthonly), NOT by class attributes — this class exists
only as a distinct trainer name so its runs get their own model dir.

KNOWN INCONSISTENCY (found in the 2026-09-27 code-structure audit): T1w and T2w produce
their val100 checkpoint via this standalone trainer + a separate training run (04_08/
04_14/04_25 wrappers), NOT via the AugLabDualVal mechanism (auglab_dualval.py) that
every other headline dataset uses to hard-link val000+val100 from ONE run. dwi_ap (see
04_41_train_dwi_ap_auglabAug_v26_6_2_dualval.sh) uses DualVal only and does NOT have
this standalone class's counterpart — so T1w/T2w and dwi_ap don't share one convention.
Not unified here deliberately: doing so would require retraining T1w/T2w's val100 arm,
which is out of scope for a structural fix.
"""
from __future__ import annotations

from on_harmony.trainers.v26_6_2 import nnUNetTrainerOnHarmonyV26_6_2


class nnUNetTrainerOnHarmonyV26_6_2_train050_val100(nnUNetTrainerOnHarmonyV26_6_2):
    """V26_6_2 for ON-Harmony (50% train / 100% val synth — set via config JSON)."""
