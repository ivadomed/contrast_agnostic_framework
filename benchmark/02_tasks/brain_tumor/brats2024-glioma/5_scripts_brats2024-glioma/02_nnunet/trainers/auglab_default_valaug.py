"""
nnUNetTrainerBraTS2024GliomaAugLabDefaultValAug — AugLab default config, but with
the GPU augmentation pipeline ALSO applied during validation at 100%.

Identical to nnUNetTrainerBraTS2024GliomaAugLabDefault (same default paper config,
same train_step, same WandB panels) EXCEPT validation no longer runs on clean data:
the same self.transforms used in train_step is applied to 100% of validation
batches, so the val distribution matches training under domain randomization
(mirroring V26_6_2). This removes the train/val distribution mismatch that the
paper-faithful trainer leaves in place, and makes the val pseudo-dice curve and
checkpoint_best selection reflect training-distribution performance.

MRO: ...AugLabDefaultValAug → ...AugLabDefault → nnUNetTrainerBraTS2024GliomaBase
     → nnUNetTrainerFast → nnUNetTrainerDAExtGPU → nnUNetTrainer
"""
from __future__ import annotations

import torch
from torch import autocast

from auglab.trainers.utils import DownsampleSegForDSTransformCustom
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import dummy_context

from brats2024_glioma.trainers.auglab_default import nnUNetTrainerBraTS2024GliomaAugLabDefault


class nnUNetTrainerBraTS2024GliomaAugLabDefaultValAug(nnUNetTrainerBraTS2024GliomaAugLabDefault):
    """AugLab default config with validation augmentation applied at 100%."""

    # Validation augments at 100% → val WandB panel shows the augmented input too
    # (inherited _log_wandb_images reads this flag).
    validation_uses_augmentation: bool = True

    @staticmethod
    def get_validation_transforms(deep_supervision_scales, is_cascaded: bool = False,
                                  foreground_labels=None, regions=None, ignore_label=None):
        """Validation transforms WITHOUT DS downsampling.

        DS downsampling is applied on the GPU in validation_step (after
        self.transforms), exactly as the base class does for training, so the
        val target arrives as a single tensor that self.transforms can augment.
        """
        return ComposeTransforms([RemoveLabelTansform(-1, 0)])

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        tgt = batch["target"]
        # get_validation_transforms (above) does not apply DS → single tensor.
        target = (tgt[0] if isinstance(tgt, (list, tuple)) else tgt).to(self.device, non_blocking=True)

        ctx = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context()
        with ctx:
            # Same GPU augmentation pipeline as train_step, applied to 100% of
            # validation batches so train/val distributions match.
            data, target = self.transforms(data, target)

            # Full-resolution augmented target kept for the pseudo-dice metric;
            # the DS list (loss only) is built from it on the GPU.
            target_full = target
            deep_supervision_scales = self._get_deep_supervision_scales()
            if deep_supervision_scales is not None:
                target_ds = DownsampleSegForDSTransformCustom(ds_scales=deep_supervision_scales)(target)
            else:
                target_ds = target

            output = self.network(data)
            del data
            l = self.loss(output, target_ds)

        # Online pseudo-dice on the highest-resolution head vs the augmented target.
        output_hr = output[0] if self.enable_deep_supervision else output
        target_hr = target_full

        axes = [0] + list(range(2, output_hr.ndim))
        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output_hr) > 0.5).long()
        else:
            output_seg = output_hr.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(
                output_hr.shape, device=output_hr.device, dtype=torch.float16)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target_hr != self.label_manager.ignore_label).float()
                target_hr = target_hr.clone()
                target_hr[target_hr == self.label_manager.ignore_label] = 0
            else:
                mask = (1 - target_hr[:, -1:]) if target_hr.dtype != torch.bool else (~target_hr[:, -1:])
                target_hr = target_hr[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target_hr, axes=axes, mask=mask)
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard, fp_hard, fn_hard = tp_hard[1:], fp_hard[1:], fn_hard[1:]

        return {"loss": l.detach().cpu().numpy(),
                "tp_hard": tp_hard, "fp_hard": fp_hard, "fn_hard": fn_hard}
