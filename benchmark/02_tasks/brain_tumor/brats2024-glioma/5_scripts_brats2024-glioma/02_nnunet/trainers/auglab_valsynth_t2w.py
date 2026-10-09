"""
nnUNetTrainerBraTS2024GliomaT2wAugLabValSynth — AugLab+ValSynth for BraTS 2024 T2w.

Thin T2w binding of nnUNetTrainerBraTS2024GliomaAugLabValSynth. Train pipeline applies
full AugLab augs + GPU synthesis (AUGLAB_PARAMS_GPU_JSON); validation pipeline applies
synth-only (AUGLAB_VAL_PARAMS_GPU_JSON). Used by auglabAug+v26_6_2 T2w training.
Uses Dataset052_BraTS2024GliomaT2w.

MRO: nnUNetTrainerBraTS2024GliomaT2wAugLabValSynth
  → nnUNetTrainerBraTS2024GliomaAugLabValSynth
  → nnUNetTrainerBraTS2024GliomaAugLabDefault
  → nnUNetTrainerBraTS2024GliomaBase
  → nnUNetTrainerFast / nnUNetTrainerDAExtGPU / nnUNetTrainer
"""
from __future__ import annotations

from brats2024_glioma.trainers.auglab_valsynth import nnUNetTrainerBraTS2024GliomaAugLabValSynth


class nnUNetTrainerBraTS2024GliomaT2wAugLabValSynth(nnUNetTrainerBraTS2024GliomaAugLabValSynth):
    """AugLab+ValSynth for T2w — same logic as T1n, different dataset + trainer name."""
    pass
