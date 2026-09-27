"""
nnUNetTrainerBraTS2024GliomaT2fAugLabValSynth — AugLab+ValSynth for BraTS 2024 T2f/FLAIR.

Thin T2f binding of nnUNetTrainerBraTS2024GliomaAugLabValSynth. Train pipeline applies
full AugLab augs + GPU synthesis (AUGLAB_PARAMS_GPU_JSON); validation pipeline applies
synth-only (AUGLAB_VAL_PARAMS_GPU_JSON). Used by auglabAug+v26_6_2 T2f/FLAIR training.
Uses Dataset053_BraTS2024GliomaT2f.

MRO: nnUNetTrainerBraTS2024GliomaT2fAugLabValSynth
  → nnUNetTrainerBraTS2024GliomaAugLabValSynth
  → nnUNetTrainerBraTS2024GliomaAugLabDefault
  → nnUNetTrainerBraTS2024GliomaBase
  → nnUNetTrainerFast / nnUNetTrainerDAExtGPU / nnUNetTrainer
"""
from __future__ import annotations

from brats2024_glioma.trainers.auglab_valsynth import nnUNetTrainerBraTS2024GliomaAugLabValSynth


class nnUNetTrainerBraTS2024GliomaT2fAugLabValSynth(nnUNetTrainerBraTS2024GliomaAugLabValSynth):
    """AugLab+ValSynth for T2f/FLAIR — same logic as T1n, different dataset + trainer name."""
    pass
