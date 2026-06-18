"""
nnUNetTrainerCHAOSAugLabV26_6_2 — AugLab GPU augmentation trainer for CHAOS used by
the auglabAug_v26_6_2_train050_val000 run (AugLab default augs + V26_6_2 GPU
transform; train synth prob set by the config, e.g. 0.5).

Differs from nnUNetTrainerCHAOSAugLabDefault ONLY by validation_uses_augmentation=True,
which affects the WandB *visualization panel* alone (it renders augmented val images).
Actual validation runs on clean, un-augmented data via stock nnUNet validation_step —
so the effective validation synth probability is 0% (hence the _val000 run name).
"""
from __future__ import annotations

from chaos.trainers.auglab_default import nnUNetTrainerCHAOSAugLabDefault


class nnUNetTrainerCHAOSAugLabV26_6_2(nnUNetTrainerCHAOSAugLabDefault):
    """AugLab for CHAOS; augmented val images shown in WandB panel (val metrics stay clean)."""

    validation_uses_augmentation: bool = True
