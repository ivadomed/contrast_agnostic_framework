"""
nnUNetTrainerOnHarmonyAugLabDefault — AugLab GPU augmentation trainer for ON-Harmony T1w.

Also used for the SynthSeg variants (noEM, EM) by pointing AUGLAB_PARAMS_GPU_JSON
at the corresponding SynthSeg JSON — there is no separate SynthSeg trainer class,
exactly as in brats2024-glioma and CHAOS.

MRO: nnUNetTrainerOnHarmonyAugLabDefault
  → nnUNetTrainerOnHarmonyBase  (do_split — anti-contamination guard)
  → nnUNetTrainerFast            (seed, epochs, WandB hooks)
  → nnUNetTrainerDAExtGPU        (GPU transforms, train_step)
  → nnUNetTrainer
"""
from __future__ import annotations

import importlib.resources
import os

import torch

import auglab.configs as _auglab_configs
from auglab.trainers.nnUNetTrainerDAExt import nnUNetTrainerDAExtGPU

from on_harmony.trainers.base import nnUNetTrainerOnHarmonyBase

_DEFAULT_CONFIG = str(
    importlib.resources.files(_auglab_configs) / "transform_params_gpu_default01-23.json"
)


class nnUNetTrainerOnHarmonyAugLabDefault(nnUNetTrainerOnHarmonyBase, nnUNetTrainerDAExtGPU):
    """
    AugLab GPU augmentation trainer for ON-Harmony T1w segmentation.

    GPU spatial + intensity transforms applied in train_step; validation runs on
    clean (un-augmented) data, paper-faithful to upstream nnUNetTrainerDAExtGPU.
    AUGLAB_PARAMS_GPU_JSON env var selects the config (default, SynthSeg, or EM).
    """

    validation_uses_augmentation: bool = False

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        if "AUGLAB_PARAMS_GPU_JSON" not in os.environ:
            os.environ["AUGLAB_PARAMS_GPU_JSON"] = _DEFAULT_CONFIG
        super().__init__(plans, configuration, fold, dataset_json, device)
