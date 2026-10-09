"""
nnUNetTrainerOnHarmonyAugLabValSynth — AugLab trainer with synthesis-only validation.

Unlike auglab_default (validation on clean data), this trainer applies a synth-only
transform during validation so val metrics reflect performance on synthesised images.

Two configs, both set by the 04_train wrapper script:
  AUGLAB_PARAMS_GPU_JSON      — TRAIN pipeline (full augmentation + synth at train prob)
  AUGLAB_VAL_PARAMS_GPU_JSON  — VAL pipeline: synth-only (all other augs off)

Used for auglabAug_v26_6_2_train025_val100 and similar experiments.
"""
from __future__ import annotations

import os

import torch
from torch import autocast

from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform

from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

from auglab.transforms.gpu.transforms import AugTransformsGPU
from auglab.trainers.utils import DownsampleSegForDSTransformCustom

from on_harmony.trainers.auglab_default import nnUNetTrainerOnHarmonyAugLabDefault


class nnUNetTrainerOnHarmonyAugLabValSynth(nnUNetTrainerOnHarmonyAugLabDefault):
    """AugLab for ON-Harmony with synth-only validation (val metrics on synthesised images)."""

    validation_uses_augmentation: bool = True

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        val_json = os.environ.get("AUGLAB_VAL_PARAMS_GPU_JSON")
        if not val_json:
            raise RuntimeError(
                "AUGLAB_VAL_PARAMS_GPU_JSON must be set (synth-only val config) for "
                "nnUNetTrainerOnHarmonyAugLabValSynth."
            )
        self.val_transforms_synth = AugTransformsGPU(json_path=val_json).to(self.device)
        print(f"[AugLabValSynth] val synth-only transforms: {val_json}")

    def get_validation_transforms(self, *args, **kwargs):
        return ComposeTransforms([RemoveLabelTansform(-1, 0)])

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = (target[0] if isinstance(target, (list, tuple)) else target).to(
            self.device, non_blocking=True
        )

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            data, target = self.val_transforms_synth(data, target)
            ds_scales = self._get_deep_supervision_scales()
            if ds_scales is not None:
                target = DownsampleSegForDSTransformCustom(ds_scales=ds_scales)(target)
            output = self.network(data)
            del data
            l = self.loss(output, target)

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float16)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = (~target[:, -1:]) if target.dtype == torch.bool else (1 - target[:, -1:])
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard, fp_hard, fn_hard = tp_hard[1:], fp_hard[1:], fn_hard[1:]
        return {"loss": l.detach().cpu().numpy(), "tp_hard": tp_hard, "fp_hard": fp_hard, "fn_hard": fn_hard}
