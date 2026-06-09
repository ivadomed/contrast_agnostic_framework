"""
nnUNetTrainerBraTS2024GliomaV26_6 — V26_6 on-the-fly synthesis for BraTS 2024.

Pipeline overview
-----------------
Workers (DA_WORKERS=8 processes) run the FULL augmentation pipeline:
  initial_patch_size data → SpatialTransform (rotate+scale) → patch_size
  → SimulateLowResolution → MirrorTransform → DownsampleSegForDS
  → batch["data"] at patch_size, batch["target"] as DS supervision list.

train_step (main thread, GPU):
  H2D → 90% (min-max → synthesize_batch_fast → z-score) | 10% identity → fwd/bwd.

No GPU spatial augment.  No thread pool.  No intermediate crop loop.
Validation uses raw (non-synthesised) centre-cropped patches.

MRO: nnUNetTrainerBraTS2024GliomaV26_6
  → nnUNetTrainerBraTS2024GliomaBase  (do_split — anti-contamination guard)
  → nnUNetTrainerV26_6                (WandB panels, on-harmony path)
  → nnUNetTrainerFast                  (seed, epochs, WandB hooks)
  → nnUNetTrainer
"""
from __future__ import annotations

import random
from contextlib import nullcontext

import torch
from torch.amp import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from brats2024_glioma.trainers.base import nnUNetTrainerBraTS2024GliomaBase
from src.nnunet.trainers.v26_6_base import (
    nnUNetTrainerV26_6,
    _build_synth_training_transforms,
    center_crop_pair,
)
from src.synthesis.v26_6_synthesis import synthesize_batch_fast


class nnUNetTrainerBraTS2024GliomaV26_6(nnUNetTrainerBraTS2024GliomaBase, nnUNetTrainerV26_6):
    """
    V26_6 for BraTS 2024 Glioma (Dataset051, single T1n channel).

    Workers deliver patch_size batches with DS targets; train_step synthesises
    on those patches and runs the network.  Epoch time target: <40 s.
    """

    # ── Data loading ───────────────────────────────────────────────────────────

    def get_dataloaders(self):
        """Configure V26_6 params and delegate to nnUNet's standard multi-process loader."""
        self._configure_v26_6()
        return nnUNetTrainer.get_dataloaders(self)

    def get_training_transforms(self, *args, **kwargs):
        """
        Return the full augmentation pipeline run by workers.

        Workers receive initial_patch_size data and output patch_size batches:
          SpatialTransform (rotate + scale) → SimulateLowRes → Mirror → DS downsample.
        """
        return _build_synth_training_transforms(
            self._patch_size_cfg,
            self._rotation_for_DA,
            self._deep_supervision_scales,
            self._mirror_axes,
            self._do_dummy_2d,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            ignore_label=self.label_manager.ignore_label,
        )

    def get_validation_transforms(self, *args, **kwargs):
        """Val workers apply only RemoveLabel; centre-crop + DS happen in validation_step."""
        from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
        from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
        return ComposeTransforms([RemoveLabelTansform(-1, 0)])

    # ── Training ───────────────────────────────────────────────────────────────

    def train_step(self, batch: dict) -> dict:
        """
        Workers deliver patch_size batches (spatially augmented + DS ready).
        Main thread: 90% synthesise, 10% identity → fwd/bwd.

        The 10% identity path feeds raw z-score-normalized T1n to the network,
        matching the inference distribution produced by nnUNetv2_predict.
        """
        data = batch["data"].to(self.device, non_blocking=True)
        _tgt = batch["target"]
        target = (
            [t.to(self.device, non_blocking=True) for t in _tgt]
            if isinstance(_tgt, (list, tuple))
            else _tgt.to(self.device, non_blocking=True)
        )

        if random.random() < 0.1:
            synth_z = data  # raw z-score-normalized T1n — matches inference distribution
        else:
            B = data.shape[0]
            flat = data.reshape(B, -1)
            v_min = flat.min(dim=1).values.view(B, 1, 1, 1, 1)
            v_max = flat.max(dim=1).values.view(B, 1, 1, 1, 1)
            data_01 = ((data - v_min) / (v_max - v_min + 1e-7)).clamp(0, 1)
            synth_z, _ = synthesize_batch_fast(data_01)

        self.optimizer.zero_grad(set_to_none=True)
        ctx = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else nullcontext()
        with ctx:
            output = self.network(synth_z)
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {"loss": l.detach().cpu().numpy()}

    # ── Validation ─────────────────────────────────────────────────────────────

    def validation_step(self, batch: dict) -> dict:
        """Validate on raw (non-synthesised) centre-cropped patches."""
        from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

        data = batch["data"].to(self.device, non_blocking=True)
        _tgt = batch["target"]
        target_raw = (
            (_tgt[0] if isinstance(_tgt, (list, tuple)) else _tgt)
            .to(self.device, non_blocking=True)
        )

        patches, patches_seg = center_crop_pair(data, target_raw, self._patch_size_cfg, n_crops=1)
        t = self._val_transforms(
            image=patches.cpu().float()[0],
            segmentation=patches_seg.cpu().to(torch.int16)[0],
        )
        data_aug = t["image"].unsqueeze(0).to(self.device, non_blocking=True)
        seg_raw = t["segmentation"]
        if isinstance(seg_raw, (list, tuple)):
            seg_aug = [s.unsqueeze(0).to(self.device, non_blocking=True) for s in seg_raw]
        else:
            seg_aug = seg_raw.unsqueeze(0).to(self.device, non_blocking=True)

        ctx = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else nullcontext()
        with ctx:
            output = self.network(data_aug)
            l = self.loss(output, seg_aug)

        output_primary = output[0] if isinstance(output, (list, tuple)) else output
        target_primary = seg_aug[0] if isinstance(seg_aug, (list, tuple)) else seg_aug

        axes = [0] + list(range(2, output_primary.ndim))
        output_seg = output_primary.argmax(1, keepdim=True)
        predicted_onehot = torch.zeros(
            output_primary.shape, device=output_primary.device, dtype=torch.float16
        )
        predicted_onehot.scatter_(1, output_seg, 1)
        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_onehot, target_primary, axes=axes)

        return {
            "loss":     l.detach().cpu().numpy(),
            "tp_hard":  tp.detach().cpu().numpy()[1:],
            "fp_hard":  fp.detach().cpu().numpy()[1:],
            "fn_hard":  fn.detach().cpu().numpy()[1:],
        }
