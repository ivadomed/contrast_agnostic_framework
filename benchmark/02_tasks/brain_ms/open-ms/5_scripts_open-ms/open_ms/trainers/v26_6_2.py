"""
nnUNetTrainerOpenMSV26_6_2 — V26_6_2 contrast synthesis for open-ms (MS lesion).

Runs the AugLab GPU ImageContrastV26_6_2GPUTransform (K-means/Voronoi contrast
synthesis, per-patch) — the same backend as CHAOS / BraTS / ON-Harmony. Synthesis
probabilities come from the config JSONs the 04_train wrapper exports (NOT class
attributes):
  AUGLAB_PARAMS_GPU_JSON      → train config (v26-only spatialDA, or full auglabAug+v26)
  AUGLAB_VAL_PARAMS_GPU_JSON  → val: transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json

MRO: nnUNetTrainerOpenMSV26_6_2
  → nnUNetTrainerOpenMSAugLabValSynth (synth-only validation)
  → nnUNetTrainerOpenMSAugLabDefault  (AugLab GPU train transforms)
  → nnUNetTrainerOpenMSBase           (do_split guard)
  → nnUNetTrainerDAExtGPU
"""
from __future__ import annotations

from open_ms.trainers.auglab_valsynth import nnUNetTrainerOpenMSAugLabValSynth


class nnUNetTrainerOpenMSV26_6_2(nnUNetTrainerOpenMSAugLabValSynth):
    """V26_6_2 for open-ms via the AugLab contrast transform (see module docstring)."""
