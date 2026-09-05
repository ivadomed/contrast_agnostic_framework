"""
nnUNetTrainerAtlasLiverHCCAugLabDefault — AugLab GPU augmentation trainer for
atlas-liver-hcc.

Also used for the SynthSeg variants (noEM, EM) and srcsm by pointing
AUGLAB_PARAMS_GPU_JSON at the corresponding JSON — there is no separate SynthSeg/srcsm
trainer class, exactly as in CHAOS / brats2024-glioma / ON-Harmony / open-ms.

MRO: nnUNetTrainerAtlasLiverHCCAugLabDefault
  -> nnUNetTrainerAtlasLiverHCCBase  (do_split — anti-contamination guard)
  -> nnUNetTrainerFast                (seed, epochs, WandB hooks)
  -> nnUNetTrainerDAExtGPU             (GPU transforms, train_step)
  -> nnUNetTrainer
"""
from __future__ import annotations

import importlib.resources
import os

import torch

import auglab.configs as _auglab_configs
from auglab.trainers.nnUNetTrainerDAExt import nnUNetTrainerDAExtGPU

from atlas_liver_hcc.trainers.base import nnUNetTrainerAtlasLiverHCCBase

_DEFAULT_CONFIG = str(
    importlib.resources.files(_auglab_configs) / "transform_params_gpu_default01-23.json"
)


class nnUNetTrainerAtlasLiverHCCAugLabDefault(nnUNetTrainerAtlasLiverHCCBase, nnUNetTrainerDAExtGPU):
    """AugLab GPU augmentation trainer for atlas-liver-hcc (HCC liver tumour) segmentation.

    GPU spatial + intensity transforms in train_step; validation on clean data.
    AUGLAB_PARAMS_GPU_JSON selects the config (default / SynthSeg / SynthSeg-EM / srcsm /
    v26_6_2 synthesis)."""

    validation_uses_augmentation: bool = False

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        if "AUGLAB_PARAMS_GPU_JSON" not in os.environ:
            os.environ["AUGLAB_PARAMS_GPU_JSON"] = _DEFAULT_CONFIG
        super().__init__(plans, configuration, fold, dataset_json, device)
