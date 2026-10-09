"""
nnUNetTrainerV26_6 — V26_6 synthesis method base.  Dataset-agnostic.

Synthesis probability — single source of truth
----------------------------------------------
Two attributes govern how often a batch is synthesised, and they are the ONLY
place the behaviour is defined.  Every train/val step and the WandB panel read
them; nothing hardcodes a probability:

    train_synth_prob : float   # P(synthesise) per training step
    val_synth_prob   : float   # P(synthesise) per validation step

Default 0.9 / 1.0: training mixes 10% real (z-scored) batches to stay close to
the inference distribution; validation is fully synthetic so the val metric
measures the synthesised distribution the network is actually trained on.
Everything fed to the network is z-score normalised (synth output is z-scored;
the raw branch z-scores per sample).

Two data pipelines share this logic:
  * full-volume (this class' train_step/validation_step): GPU synth on the whole
    volume → crop → light transforms.  Used by on-harmony / plain runs.
  * patch-based (_patch_train_step / _patch_validation_step): nnUNet's default
    loader produces augmented patches; we synthesise the patch.  Used by the
    dataset trainers (CHAOS, BraTS) via two thin overrides.

The synthesis variant is chosen by ONE hook, _synthesize(), so train and val can
never diverge: V26_6 = whole-image; V26_6_2 overrides it for the per-label remap.

WandB panels reflect REALITY: train/val steps stash the exact tensor fed to the
network (+ whether synth was applied); _log_wandb_images renders that stash and
never re-synthesises for display.

Dataset-specific subclasses override:
  get_dataloaders()        — provide custom loaders (e.g. ON-Harmony RAM cache)
  train/validation_step    — call _patch_train_step/_patch_validation_step
  _wandb_slice_axis         — 0 for axial (anisotropic) datasets, -1 otherwise

Transform builder functions (_build_v26_fast_transforms, etc.) are module-level
so that SynthSeg base can import and reuse them.
"""
from __future__ import annotations

import os
import random
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

    # ── Synthesis probability — single source of truth (see module docstring) ──
    train_synth_prob: float = 0.9
    val_synth_prob: float = 1.0

    # WandB panel slice axis: -1 = last spatial axis (isotropic), 0 = axial first
    # spatial axis (anisotropic abdominal volumes).  Dataset subclasses may set 0.
    _wandb_slice_axis: int = -1

    # ── Initialize ────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        super().initialize()

        sys.path.insert(0, str(_PROJECT_ROOT))
        from src.synthesis.histogram_ops import DifferentiableHistogram3D
        from src.synthesis.target_generators import V26_6SignedAlphaTargetGenerator

        # GPU generator + hist module kept for WandB image logging panels.
        self._generator = V26_6SignedAlphaTargetGenerator()
        self._hist_module = DifferentiableHistogram3D(64).to(self.device)

        # Honest-viz stash: the exact tensor each step fed to the network.
        self._viz_cache: dict = {}

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
        self._prev_synth_flag = None  # full-volume prefetch: flag of forwarded batch

    # ── Default data loaders (nnUNet standard) ────────────────────────────────

    def get_dataloaders(self):
        """Use nnUNet's default loaders.  Override in dataset-specific subclasses."""
        self._setup_v26_training()
        return super().get_dataloaders()

    def get_training_transforms(self, *args, **kwargs):
        return self._train_transforms

    # ── Synthesis primitives — shared by every step and the WandB panel ────────

    @staticmethod
    def _minmax01(data: torch.Tensor) -> torch.Tensor:
        """Per-sample min-max → [0, 1] (synthesis input range)."""
        B = data.shape[0]
        flat = data.reshape(B, -1)
        v_min = flat.min(dim=1).values.view(B, 1, 1, 1, 1)
        v_max = flat.max(dim=1).values.view(B, 1, 1, 1, 1)
        return ((data - v_min) / (v_max - v_min + 1e-7)).clamp(0, 1)

    @staticmethod
    def _zscore(vol: torch.Tensor) -> torch.Tensor:
        """Per-sample z-score (what the network sees for the real/raw branch)."""
        B = vol.shape[0]
        flat = vol.reshape(B, -1)
        mean = flat.mean(dim=1).view(B, 1, 1, 1, 1)
        std = flat.std(dim=1).view(B, 1, 1, 1, 1)
        return (vol - mean) / (std + 1e-7)

    def _synthesize(self, data01: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        """V26_6 whole-image synthesis → z-scored volume.

        THE synthesis hook: V26_6_2 overrides this for the per-label remap, so
        train and val always use the same variant.  `seg` is unused here (kept
        for a uniform signature).
        """
        synth_z, _ = synthesize_batch_fast(data01)
        return synth_z

    def _synth_or_raw(self, data: torch.Tensor, seg: torch.Tensor,
                      synth_prob: float) -> tuple[torch.Tensor, bool, torch.Tensor]:
        """Coin-flip on synth_prob.  Returns (net_input, synth_applied, src01).

        `data` is a z-scored tensor.  Synth branch: min-max → _synthesize → z.
        Raw branch: return `data` unchanged (already z-scored).  Either way the
        output is z-score normalised — nothing un-normalised reaches the network.
        `src01` is the min-max [0,1] original (the un-synthesised sample), kept so
        the WandB panel can show the real source / recompute the synth pipeline.
        """
        src01 = self._minmax01(data)
        if random.random() < synth_prob:
            return self._synthesize(src01, seg), True, src01
        return data, False, src01

    def _stash_viz(self, tag: str, net_in: torch.Tensor, seg: torch.Tensor,
                   synth_applied: bool, src01: torch.Tensor | None = None) -> None:
        """Record the EXACT tensor fed to the network (+ the [0,1] original) for
        honest WandB panels.  src01 is None on the full-volume path (the cropped
        source isn't aligned), which the panel hook handles gracefully."""
        seg0 = (seg[0] if isinstance(seg, (list, tuple)) else seg)
        self._viz_cache[tag] = {
            "net_in": net_in[0:1].detach(),
            "gt": seg0[0:1].detach(),
            "synth": bool(synth_applied),
            "src01": None if src01 is None else src01[0:1].detach(),
        }

    def _val_metrics(self, l: torch.Tensor, output, target) -> dict:
        """Standard nnUNet hard tp/fp/fn for the val Dice."""
        from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
        out = output[0] if isinstance(output, (list, tuple)) else output
        tgt = target[0] if isinstance(target, (list, tuple)) else target
        axes = [0] + list(range(2, out.ndim))
        seg = out.argmax(1, keepdim=True)
        oh = torch.zeros(out.shape, device=out.device, dtype=torch.float16)
        oh.scatter_(1, seg, 1)
        tp, fp, fn, _ = get_tp_fp_fn_tn(oh, tgt, axes=axes)
        return {
            "loss":    l.detach().cpu().numpy(),
            "tp_hard": tp.detach().cpu().numpy()[1:],
            "fp_hard": fp.detach().cpu().numpy()[1:],
            "fn_hard": fn.detach().cpu().numpy()[1:],
        }

    # ── Patch pipeline (nnUNet default loader) — for dataset trainers ──────────

    def _patch_train_step(self, batch: dict) -> dict:
        """Train step for the patch pipeline: synthesise the (already cropped +
        augmented) patch with train_synth_prob, forward/backward.  Dataset
        trainers call this from a one-line train_step override."""
        data = batch["data"].to(self.device, non_blocking=True)
        _tgt = batch["target"]
        target = (
            [t.to(self.device, non_blocking=True) for t in _tgt]
            if isinstance(_tgt, (list, tuple))
            else _tgt.to(self.device, non_blocking=True)
        )
        seg0 = (target[0] if isinstance(target, (list, tuple)) else target).long()

        net_in, synth_applied, src01 = self._synth_or_raw(data, seg0, self.train_synth_prob)
        self._stash_viz("train", net_in, seg0, synth_applied, src01)

        self.optimizer.zero_grad(set_to_none=True)
        ctx = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else nullcontext()
        with ctx:
            output = self.network(net_in)
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

    def _patch_validation_step(self, batch: dict) -> dict:
        """Validation step for the patch pipeline: centre-crop → synthesise with
        val_synth_prob → val transforms (DS) → forward → metrics."""
        data = batch["data"].to(self.device, non_blocking=True)
        _tgt = batch["target"]
        seg = (
            (_tgt[0] if isinstance(_tgt, (list, tuple)) else _tgt)
            .to(self.device, non_blocking=True)
        )

        patch, patch_seg = center_crop_pair(data, seg, self._patch_size_cfg, n_crops=1)
        net_in, synth_applied, src01 = self._synth_or_raw(patch, patch_seg.long(), self.val_synth_prob)

        t = self._val_transforms(
            image=net_in.cpu().float()[0],
            segmentation=patch_seg.cpu().to(torch.int16)[0],
        )
        data_aug = t["image"].unsqueeze(0).to(self.device, non_blocking=True)
        seg_raw = t["segmentation"]
        if isinstance(seg_raw, (list, tuple)):
            seg_aug = [s.unsqueeze(0).to(self.device, non_blocking=True) for s in seg_raw]
        else:
            seg_aug = seg_raw.unsqueeze(0).to(self.device, non_blocking=True)

        self._stash_viz("val", data_aug, seg_aug, synth_applied, src01)

        ctx = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else nullcontext()
        with ctx:
            output = self.network(data_aug)
            l = self.loss(output, seg_aug)
        return self._val_metrics(l, output, seg_aug)

    # ── Full-volume pipeline (on-harmony / plain) ──────────────────────────────

    def train_step(self, batch: dict) -> dict:
        """
        Full-volume GPU synthesis → random crop → background transforms → fwd/bwd.

        synthesise (train_synth_prob) or z-score the real volume, then crop.
        While GPU runs forward/backward, background threads transform the next
        batch.  Stashes the forwarded tensor (+ its synth flag) for honest viz.
        """
        data = batch["data"].to(self.device, non_blocking=True)
        _tgt = batch["target"]
        target = (_tgt[0] if isinstance(_tgt, (list, tuple)) else _tgt).to(self.device, non_blocking=True)

        if self._prefetch_future is not None:
            data_list, seg_list = self._prefetch_future.result()
        else:
            data_list, seg_list = None, None  # first step

        B = data.shape[0]
        data_01 = self._minmax01(data)
        data_01, target = gpu_spatial_augment(data_01, target)
        if random.random() < self.train_synth_prob:
            net_vol = self._synthesize(data_01, target)
            synth_applied = True
        else:
            net_vol = self._zscore(data_01)
            synth_applied = False

        patch_list, seg_list_p = [], []
        for i in range(B):
            p, ps = random_crop_pair(net_vol[i:i+1], target[i:i+1], self._patch_size_cfg, n_crops=1)
            patch_list.append(p)
            seg_list_p.append(ps)
        n_patches = B
        patches = torch.cat(patch_list, dim=0)
        patches_cpu = patches.cpu().float()
        patches_seg_cpu = torch.cat(seg_list_p, dim=0).cpu().to(torch.int16)

        def _do_transforms(patchc, segc, n):
            outs = [self._train_transforms(image=patchc[b], segmentation=segc[b]) for b in range(n)]
            return [o["image"] for o in outs], [o["segmentation"] for o in outs]

        self._prefetch_future = self._transform_pool.submit(
            _do_transforms, patches_cpu, patches_seg_cpu, n_patches
        )

        if data_list is None:
            data_list, seg_list = self._prefetch_future.result()
            self._prefetch_future = None
            forwarded_flag = synth_applied  # forwarding the batch we just made
        else:
            forwarded_flag = self._prev_synth_flag  # forwarding the previous batch
        self._prev_synth_flag = synth_applied

        data_aug = torch.stack(data_list).to(self.device, non_blocking=True)
        if isinstance(seg_list[0], (list, tuple)):
            n_scales = len(seg_list[0])
            seg_aug = [
                torch.stack([seg_list[b][s] for b in range(n_patches)]).to(self.device, non_blocking=True)
                for s in range(n_scales)
            ]
        else:
            seg_aug = torch.stack(seg_list).to(self.device, non_blocking=True)

        self._stash_viz("train", data_aug, seg_aug,
                        synth_applied if forwarded_flag is None else forwarded_flag,
                        data_01)  # src01 for full-volume path (min-max original)

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

    def validation_step(self, batch: dict) -> dict:
        """Full-volume val: synthesise (val_synth_prob) or z-score → centre-crop →
        val transforms → forward → metrics.  Same _synthesize hook as train."""
        data = batch["data"].to(self.device, non_blocking=True)
        _tgt = batch["target"]
        target = (_tgt[0] if isinstance(_tgt, (list, tuple)) else _tgt).to(self.device, non_blocking=True)

        data_01 = self._minmax01(data)
        if random.random() < self.val_synth_prob:
            net_vol = self._synthesize(data_01, target)
            synth_applied = True
        else:
            net_vol = self._zscore(data_01)
            synth_applied = False

        patches, patches_seg = center_crop_pair(net_vol, target, self._patch_size_cfg, n_crops=1)
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

        self._stash_viz("val", data_aug, seg_aug, synth_applied, data_01)

        ctx = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else nullcontext()
        with ctx:
            output = self.network(data_aug)
            l = self.loss(output, seg_aug)
        return self._val_metrics(l, output, seg_aug)

    # ── WandB image logging — renders the real stashed input (never re-synth) ──

    def _slice2d(self, vol3d: np.ndarray) -> np.ndarray:
        """Mid-slice along _wandb_slice_axis (0 = axial first axis, -1 = last)."""
        ax = self._wandb_slice_axis % vol3d.ndim
        mid = vol3d.shape[ax] // 2
        return np.take(vol3d, mid, axis=ax)

    @staticmethod
    def _disp(v: np.ndarray) -> np.ndarray:
        """Robust [0,1] display normalisation (2-98 percentile over non-zero)."""
        b = v[v != 0]
        if len(b) == 0:
            return np.zeros_like(v)
        lo, hi = np.percentile(b, 2), np.percentile(b, 98)
        out = np.zeros_like(v)
        m = v != 0
        out[m] = np.clip((v[m] - lo) / max(hi - lo, 1e-6), 0, 1)
        return out

    def _viz_prefix_panels(self, c: dict) -> list:
        """Extra panels shown BEFORE [net input | GT | prediction].

        Returns a list of (title, image2d, cmap, vmin, vmax).  Base behaviour:
        show the original (pre-synth) sample as reference when available.
        V26_6_2 overrides this to add the K-means parcellation + whole-image
        synth stages — always honest: a raw step shows the original, never synth.
        """
        src01 = c.get("src01")
        if src01 is None:
            return []
        t1w = self._slice2d(self._disp(src01[0, 0].cpu().float().numpy()))
        return [("original (pre-synth)", t1w, "gray", None, None)]

    def _log_wandb_images(self, epoch: int = 0) -> None:
        """HONEST per-split panel: [prefix…] | network input | GT | prediction.

        The input panel is the EXACT tensor the last train/val step fed to the
        network (stashed in _viz_cache) — synth or raw, per *_synth_prob.  No
        synthesis is run for display, so the picture can never disagree with the
        network's reality; a raw step is shown as the real (raw) sample.
        """
        try:
            import wandb
            if wandb.run is None or not getattr(self, "_viz_cache", None):
                return
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            def to_np(t): return t.detach().cpu().float().numpy()

            probs = {"train": self.train_synth_prob, "val": self.val_synth_prob}
            name = type(self).__name__
            log_dict = {}
            for tag in ("train", "val"):
                c = self._viz_cache.get(tag)
                if c is None:
                    continue
                net_in = c["net_in"].to(self.device)
                self.network.eval()
                with torch.no_grad():
                    logits = self.network(net_in)
                logits = logits[0] if isinstance(logits, (list, tuple)) else logits
                pred = logits.argmax(1, keepdim=True)
                self.network.train()

                kind = "synth" if c["synth"] else "raw"
                panels = list(self._viz_prefix_panels(c))
                panels.append((f"net input ({kind})",
                               self._slice2d(self._disp(to_np(net_in[0, 0]))), "gray", None, None))
                panels.append(("GT seg",
                               self._slice2d(to_np(c["gt"][0, 0].float())), "tab10", 0, 6))
                panels.append(("Prediction",
                               self._slice2d(to_np(pred[0, 0].float())), "tab10", 0, 6))

                n = len(panels)
                fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
                if n == 1:
                    axes = [axes]
                for ax, (title, img, cmap, vmin, vmax) in zip(axes, panels):
                    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
                    ax.set_title(title, fontsize=8)
                    ax.axis("off")
                plt.suptitle(f"{name} {tag} (p_synth={probs[tag]:.2f}) fold{self.fold} ep{epoch}", fontsize=9)
                plt.tight_layout()
                log_dict[f"{tag}/panel"] = wandb.Image(fig)
                plt.close(fig)

            if not log_dict:
                return
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
