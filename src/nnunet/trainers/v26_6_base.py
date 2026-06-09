"""
nnUNetTrainerV26_6 — V26_6 synthesis method base.  Dataset-agnostic.

train_step pipeline
-------------------
  1. Full-volume min-max norm → [0, 1]
  2. GPU affine augmentation (rotation + scaling) via gpu_spatial_augment  (~30 ms)
  3. synthesize_volume_fast: K-means parcellation + signed-alpha remap    (~50 ms)
  4. Random crop to patch_size
  5. Background threads: Mirror + SimulateLowRes + DS downsample           (~20 ms)
  6. H2D → forward / backward

validation_step uses the same synthesize_volume_fast path → no train/val
distribution mismatch.

Dataset-specific subclasses override:
  get_dataloaders()        — provide custom loaders (e.g. ON-Harmony RAM cache)
  _log_wandb_images()      — log sample panels with dataset-specific data

Transform builder functions (_build_v26_fast_transforms, etc.) are module-level
so that SynthSeg base can import and reuse them.
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast

from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform

from src.nnunet.trainers.fast import BASE_SEED, nnUNetTrainerFast
from src.nnunet.transforms.synth_aug import center_crop_pair, random_crop_pair
from src.synthesis.v26_6_synthesis import synthesize_volume_fast, synthesize_batch_fast, gpu_spatial_augment

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class nnUNetTrainerV26_6(nnUNetTrainerFast):
    """
    V26_6 on-the-fly GPU synthesis trainer.  Works with any nnUNet dataset.

    Dataset-specific subclasses override get_dataloaders() to provide custom
    loaders (e.g. ON-Harmony RAM cache).  WandB image logging is implemented
    here using the generic self.dataloader_train / self.dataloader_val
    attributes — no dataset knowledge required.
    """

    synth_prob: float = 1.0

    # ── Initialize ────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        super().initialize()

        sys.path.insert(0, str(_PROJECT_ROOT))
        from src.synthesis.histogram_ops import DifferentiableHistogram3D
        from src.synthesis.target_generators import V26_6SignedAlphaTargetGenerator

        # GPU generator + hist module kept for WandB image logging panels.
        # Training and validation use synthesize_volume_fast() directly.
        self._generator = V26_6SignedAlphaTargetGenerator()
        self._hist_module = DifferentiableHistogram3D(64).to(self.device)

    # ── Config helpers ────────────────────────────────────────────────────────

    def _configure_v26_6(self) -> None:
        """Compute and cache V26_6 training config attributes.

        Called at the start of get_dataloaders() — both by this class and by
        dataset-specific subclasses that fully override get_dataloaders().
        """
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        deep_supervision_scales = self._get_deep_supervision_scales()
        patch_size = self.configuration_manager.patch_size

        self._initial_patch_size = tuple(int(x) for x in initial_patch_size)
        self._patch_size_cfg = tuple(int(x) for x in patch_size)
        self._rotation_for_DA = rotation_for_DA
        self._mirror_axes = mirror_axes
        self._deep_supervision_scales = deep_supervision_scales
        self._do_dummy_2d = do_dummy_2d_data_aug

        self._train_transforms = _build_v26_fast_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            ignore_label=self.label_manager.ignore_label,
        )
        self._val_transforms = _build_val_transforms(deep_supervision_scales)

    def _setup_v26_training(self) -> None:
        """Set up V26_6 config + thread pool. Call at the start of get_dataloaders()."""
        self._configure_v26_6()
        n_workers = int(os.environ.get("NNUNET_TRANSFORM_WORKERS", "4"))
        self._transform_pool = ThreadPoolExecutor(max_workers=n_workers)
        self._prefetch_future = None

    # ── Default data loaders (nnUNet standard) ────────────────────────────────

    def get_dataloaders(self):
        """Use nnUNet's default loaders.  Override in dataset-specific subclasses."""
        self._setup_v26_training()
        return super().get_dataloaders()

    def get_training_transforms(self, *args, **kwargs):
        return self._train_transforms

    # ── train_step ────────────────────────────────────────────────────────────

    def train_step(self, batch: dict) -> dict:
        """
        GPU synthesis → random crop → background transforms → forward/backward.

        While GPU runs forward/backward (~200 ms), background threads apply
        Mirror + SimulateLowRes + DS downsample to the next batch (~20 ms).
        """
        data = batch["data"].to(self.device, non_blocking=True)
        _tgt = batch["target"]
        target = (_tgt[0] if isinstance(_tgt, (list, tuple)) else _tgt).to(self.device, non_blocking=True)

        # Consume pre-computed transforms from previous step
        if self._prefetch_future is not None:
            data_list, seg_list = self._prefetch_future.result()
        else:
            data_list, seg_list = None, None  # first step

        B = data.shape[0]

        # Per-sample min-max norm to [0, 1]
        flat  = data.reshape(B, -1)
        v_min = flat.min(dim=1).values.view(B, 1, 1, 1, 1)
        v_max = flat.max(dim=1).values.view(B, 1, 1, 1, 1)
        data_01 = ((data - v_min) / (v_max - v_min + 1e-7)).clamp(0, 1)

        # Batch GPU spatial augment + batch synthesis
        data_01, target = gpu_spatial_augment(data_01, target)
        synth_z, _      = synthesize_batch_fast(data_01)

        # Per-sample random crop (different crop location per volume)
        patch_list, seg_list_p = [], []
        for i in range(B):
            p, ps = random_crop_pair(synth_z[i:i+1], target[i:i+1], self._patch_size_cfg, n_crops=1)
            patch_list.append(p)
            seg_list_p.append(ps)
        n_patches = B
        patches = torch.cat(patch_list, dim=0)
        patches_cpu = patches.cpu().float()
        patches_seg_cpu = torch.cat(seg_list_p, dim=0).cpu().to(torch.int16)

        def _do_transforms(patchc, segc, n):
            # Serial loop on a persistent pool thread (overlaps GPU fwd/bwd).
            # n is 1-2, so spawning a fresh per-step ThreadPoolExecutor (old code)
            # was pure overhead; reusing self._transform_pool avoids that churn.
            outs = [self._train_transforms(image=patchc[b], segmentation=segc[b]) for b in range(n)]
            return [o["image"] for o in outs], [o["segmentation"] for o in outs]

        self._prefetch_future = self._transform_pool.submit(
            _do_transforms, patches_cpu, patches_seg_cpu, n_patches
        )

        if data_list is None:
            data_list, seg_list = self._prefetch_future.result()
            self._prefetch_future = None

        data_aug = torch.stack(data_list).to(self.device, non_blocking=True)
        if isinstance(seg_list[0], (list, tuple)):
            n_scales = len(seg_list[0])
            seg_aug = [
                torch.stack([seg_list[b][s] for b in range(n_patches)]).to(self.device, non_blocking=True)
                for s in range(n_scales)
            ]
        else:
            seg_aug = torch.stack(seg_list).to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        ctx = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else nullcontext()
        with ctx:
            output = self.network(data_aug)
            l = self.loss(output, seg_aug)

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

    # ── validation_step ───────────────────────────────────────────────────────

    def validation_step(self, batch: dict) -> dict:
        """Same synthesize_volume_fast() as train_step → identical distribution."""
        data = batch["data"].to(self.device, non_blocking=True)
        _tgt = batch["target"]
        target = (_tgt[0] if isinstance(_tgt, (list, tuple)) else _tgt).to(self.device, non_blocking=True)

        B_val = data.shape[0]
        flat_v  = data.reshape(B_val, -1)
        v_min_v = flat_v.min(dim=1).values.view(B_val, 1, 1, 1, 1)
        v_max_v = flat_v.max(dim=1).values.view(B_val, 1, 1, 1, 1)
        image_01 = ((data - v_min_v) / (v_max_v - v_min_v + 1e-7)).clamp(0.0, 1.0)
        synth_z, _ = synthesize_batch_fast(image_01)

        patches, patches_seg = center_crop_pair(synth_z, target, self._patch_size_cfg, n_crops=1)

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

        from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
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

    # ── WandB image logging ───────────────────────────────────────────────────

    def _log_wandb_images(self, epoch: int = 0) -> None:
        """
        Log a 4-panel V26_6 image: T1w | V26_6 synth | GT seg | Prediction.

        Uses self.dataloader_train / self.dataloader_val — standard nnUNet
        attributes, so this works regardless of which loader was set by
        get_dataloaders().
        """
        try:
            import wandb
            if wandb.run is None:
                return
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            def to_np(t): return t.detach().cpu().float().numpy()

            def norm_brain(v):
                b = v[v != 0]
                if len(b) == 0: return np.zeros_like(v)
                lo, hi = np.percentile(b, 2), np.percentile(b, 98)
                out = np.zeros_like(v)
                m = v != 0
                out[m] = np.clip((v[m] - lo) / max(hi - lo, 1e-6), 0, 1)
                return out

            def make_panel(batch):
                data_gpu = batch["data"].to(self.device)
                seg_raw  = batch["target"]
                seg_gpu  = (seg_raw[0] if isinstance(seg_raw, (list, tuple)) else seg_raw).to(self.device)
                v_min = data_gpu.min(); v_max = data_gpu.max()
                img01 = ((data_gpu - v_min) / (v_max - v_min + 1e-7)).clamp(0, 1)
                with torch.no_grad():
                    synth_z, synth_01 = synthesize_volume_fast(img01)
                raw_crop, seg_crop = center_crop_pair(data_gpu,  seg_gpu, self._patch_size_cfg)
                s01_crop, _        = center_crop_pair(synth_01,  seg_gpu, self._patch_size_cfg)
                sz_crop,  _        = center_crop_pair(synth_z,   seg_gpu, self._patch_size_cfg)
                t_out = self._val_transforms(
                    image=sz_crop.cpu().float()[0],
                    segmentation=seg_crop.cpu().to(torch.int16)[0],
                )
                inp = t_out["image"].unsqueeze(0).to(self.device)
                self.network.eval()
                logits = self.network(inp)
                logits = logits[0] if isinstance(logits, (list, tuple)) else logits
                pred = logits.argmax(1, keepdim=True)
                self.network.train()
                mid = raw_crop.shape[-1] // 2
                return (
                    norm_brain(to_np(raw_crop[0, 0])[:, :, mid]),
                    np.clip(to_np(s01_crop[0, 0])[:, :, mid], 0, 1),
                    to_np(seg_crop[0, 0].float())[:, :, mid],
                    to_np(pred[0, 0].float())[:, :, mid],
                )

            log_dict = {}
            for tag, loader in [("train", self.dataloader_train), ("val", self.dataloader_val)]:
                raw, synth, gt, pred = make_panel(next(loader))
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                axes[0].imshow(raw,   cmap="gray");                  axes[0].set_title("T1w");         axes[0].axis("off")
                axes[1].imshow(synth, cmap="gray");                  axes[1].set_title("V26_6 synth"); axes[1].axis("off")
                axes[2].imshow(gt,    cmap="tab10", vmin=0, vmax=6); axes[2].set_title("GT seg");      axes[2].axis("off")
                axes[3].imshow(pred,  cmap="tab10", vmin=0, vmax=6); axes[3].set_title("Prediction");  axes[3].axis("off")
                plt.suptitle(f"v26_6 {tag} fold{self.fold} ep{epoch}", fontsize=9)
                plt.tight_layout()
                log_dict[f"{tag}/panel"] = wandb.Image(fig)
                plt.close(fig)

            # Clamp to wandb's current step so a resumed run (internal step ahead of
            # the resumed epoch) doesn't get a rejected out-of-order log.
            _safe_step = max(epoch, getattr(wandb.run, "step", epoch) or epoch)
            wandb.log(log_dict, step=_safe_step)
        except Exception as e:
            import traceback
            print(f"[WandB] V26_6 image log failed: {e}\n{traceback.format_exc()}")


# ── Transform builders ────────────────────────────────────────────────────────
# Module-level so that synthseg_base.py can import _build_v26_fast_transforms.

def _build_synth_training_transforms(
    patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
    do_dummy_2d, use_mask_for_norm=None, ignore_label=None,
) -> ComposeTransforms:
    """
    Full transforms WITH SpatialTransform — used by SynthSeg trainers.

    SynthSeg generates batches at initial_patch_size; SpatialTransform crops
    and optionally rotates to the final patch_size.
    V26_6 uses _build_v26_fast_transforms() instead (no SpatialTransform).
    """
    transforms = []

    if do_dummy_2d:
        from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform
        transforms.append(Convert3DTo2DTransform())
        patch_size_spatial = patch_size[1:]
        ignore_axes = (0,)
    else:
        patch_size_spatial = patch_size
        ignore_axes = None

    transforms.append(SpatialTransform(
        patch_size_spatial, patch_center_dist_from_border=0, random_crop=False,
        p_elastic_deform=0, p_rotation=0.2, rotation=rotation_for_DA,
        p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
        bg_style_seg_sampling=False, border_mode_seg="constant", padding_value_seg=-1,
    ))

    if do_dummy_2d:
        from batchgeneratorsv2.transforms.utils.pseudo2d import Convert2DTo3DTransform
        transforms.append(Convert2DTo3DTransform())

    transforms.append(RandomTransform(
        SimulateLowResolutionTransform(
            scale=(0.5, 1), synchronize_channels=False, synchronize_axes=True,
            ignore_axes=ignore_axes, allowed_channels=None, p_per_channel=0.5,
        ), apply_probability=0.25,
    ))

    if mirror_axes is not None and len(mirror_axes) > 0:
        transforms.append(MirrorTransform(allowed_axes=mirror_axes))

    if use_mask_for_norm is not None and any(use_mask_for_norm):
        transforms.append(MaskImageTransform(
            apply_to_channels=[i for i, v in enumerate(use_mask_for_norm) if v],
            channel_idx_in_seg=0, set_outside_to=0,
        ))

    transforms.append(RemoveLabelTansform(-1, 0))
    if deep_supervision_scales is not None:
        transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

    return ComposeTransforms(transforms)


def _build_v26_fast_transforms(
    patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
    do_dummy_2d, use_mask_for_norm=None, ignore_label=None,
) -> ComposeTransforms:
    """
    CPU transforms for V26_6 — NO SpatialTransform.

    gpu_spatial_augment() handles rotation + scaling on the full volume before
    synthesis (~30 ms GPU vs ~1500 ms CPU SpatialTransform when rotation fires).
    """
    transforms = []
    ignore_axes = (0,) if do_dummy_2d else None

    transforms.append(RandomTransform(
        SimulateLowResolutionTransform(
            scale=(0.5, 1), synchronize_channels=False, synchronize_axes=True,
            ignore_axes=ignore_axes, allowed_channels=None, p_per_channel=0.5,
        ), apply_probability=0.25,
    ))

    if mirror_axes is not None and len(mirror_axes) > 0:
        transforms.append(MirrorTransform(allowed_axes=mirror_axes))

    if use_mask_for_norm is not None and any(use_mask_for_norm):
        transforms.append(MaskImageTransform(
            apply_to_channels=[i for i, v in enumerate(use_mask_for_norm) if v],
            channel_idx_in_seg=0, set_outside_to=0,
        ))

    transforms.append(RemoveLabelTansform(-1, 0))
    if deep_supervision_scales is not None:
        transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

    return ComposeTransforms(transforms)


def _build_val_transforms(deep_supervision_scales, ignore_label=None) -> ComposeTransforms:
    transforms = [RemoveLabelTansform(-1, 0)]
    if deep_supervision_scales is not None:
        transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
    return ComposeTransforms(transforms)
