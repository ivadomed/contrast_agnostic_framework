"""
nnUNetTrainerBraTS2024GliomaV26_6_2 — V26_6_2 synthesis for BraTS 2024 Glioma.

CHANGED 2026-06-25: v26_6_2 now runs the **AugLab GPU** ImageContrastV26_6_2GPUTransform
(patch-based, standard nnU-Net pipeline) instead of the native src/synthesis path — the
SAME K-means/Voronoi contrast synthesis, applied per-patch. This makes the synthesis
backend identical across all datasets and avoids the `torch.multinomial` 2**24 limit the
native full-volume path hit on large volumes.

"Without the AugLab augmentation": the train config keeps ONLY the contrast synthesis +
standard spatial DA (rotation/scaling via nnUNetSpatialTransform, mirroring via Flip);
all other AugLab intensity transforms are off. Synth probabilities come entirely from the
config JSONs the 04_train wrapper exports (NOT class attributes):
  AUGLAB_PARAMS_GPU_JSON      → train: transform_params_gpu_v26_6_2_synth_spatialDA_train0NN.json
  AUGLAB_VAL_PARAMS_GPU_JSON  → val:   transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json

MRO: nnUNetTrainerBraTS2024GliomaV26_6_2
  → nnUNetTrainerBraTS2024GliomaAugLabValSynth  (synth-only validation)
  → nnUNetTrainerBraTS2024GliomaAugLabDefault   (AugLab GPU train transforms)
  → nnUNetTrainerBraTS2024GliomaBase            (do_split — anti-contamination guard)
  → nnUNetTrainerDAExtGPU                        (standard patch loader + GPU transforms)
"""
from __future__ import annotations

from brats2024_glioma.trainers.auglab_valsynth import nnUNetTrainerBraTS2024GliomaAugLabValSynth


class nnUNetTrainerBraTS2024GliomaV26_6_2(nnUNetTrainerBraTS2024GliomaAugLabValSynth):
    """V26_6_2 for BraTS 2024 Glioma via the AugLab contrast transform (see module docstring)."""
