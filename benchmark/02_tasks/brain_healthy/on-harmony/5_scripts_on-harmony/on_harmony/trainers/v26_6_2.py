"""
nnUNetTrainerOnHarmonyV26_6_2 — V26_6_2 trainer for the ON-Harmony dataset.

CHANGED 2026-06-25: v26_6_2 now runs the **AugLab GPU** ImageContrastV26_6_2GPUTransform
(patch-based, via the standard nnU-Net pipeline) instead of the native full-volume
src/synthesis path. It is the SAME K-means/Voronoi contrast synthesis, but applied
per-patch — which avoids the `torch.multinomial` 2**24-category limit that crashed the
native full-volume path on large ON-Harmony T2w volumes.

"Without the AugLab augmentation": the train config keeps ONLY the contrast synthesis +
standard spatial DA (rotation/scaling via nnUNetSpatialTransform, mirroring via Flip);
every other AugLab intensity transform is off. The synthesis probability is set entirely
by the config JSONs the 04_train wrapper exports (NOT by class attributes):
  AUGLAB_PARAMS_GPU_JSON      → train: transform_params_gpu_v26_6_2_synth_spatialDA_train0NN.json
  AUGLAB_VAL_PARAMS_GPU_JSON  → val:   transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json

MRO: nnUNetTrainerOnHarmonyV26_6_2
  → nnUNetTrainerOnHarmonyAugLabValSynth  (synth-only validation)
  → nnUNetTrainerOnHarmonyAugLabDefault   (AugLab GPU train transforms)
  → nnUNetTrainerOnHarmonyBase            (do_split — anti-contamination guard)
  → nnUNetTrainerDAExtGPU                 (standard patch loader + GPU transforms)
"""
from __future__ import annotations

from on_harmony.trainers.auglab_valsynth import nnUNetTrainerOnHarmonyAugLabValSynth


class nnUNetTrainerOnHarmonyV26_6_2(nnUNetTrainerOnHarmonyAugLabValSynth):
    """V26_6_2 for ON-Harmony via the AugLab contrast transform (see module docstring)."""
