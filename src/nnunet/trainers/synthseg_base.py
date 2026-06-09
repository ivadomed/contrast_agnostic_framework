"""
nnUNetTrainerSynthSeg — SynthSeg synthesis method base.  Dataset-agnostic.

Training:  BrainGenerator (TF/CPU) → n_crops=4 per call, prefetch thread.
           GPU spatial augmentation (~9ms) + SimulateLowRes + mirroring.
Validation: Real T1w from nnUNet preprocessed + SimulateLowRes only.
           No BrainGenerator during validation → unbiased model selection.

Subclasses must define:
  _labels_dir : Path  — folder of fold-specific label files for BrainGenerator
  synthseg_mode : str — "A" (uniform priors) or "B" (mix_prior_and_random)

Dataset-specific subclasses additionally override _log_wandb_images().
"""
from __future__ import annotations

import queue
import threading
from abc import abstractmethod
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast

from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform

from src.nnunet.trainers.fast import nnUNetTrainerFast
from src.nnunet.trainers.v26_6_base import _build_v26_fast_transforms
from src.nnunet.transforms.synth_aug import random_crop_pair
from src.nnunet.transforms.synthseg_aug import build_brain_generator, generate_synthseg_batch
from src.synthesis.v26_6_synthesis import gpu_spatial_augment

_N_CROPS = 4  # crops per BrainGenerator call (~2s); reduces call overhead


def _build_synthseg_val_transforms(deep_supervision_scales, ignore_label=None) -> ComposeTransforms:
    """Val transforms: SimulateLowRes (matches V26_6 resolution augmentation) + DS."""
    transforms = [
        RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1), synchronize_channels=False, synchronize_axes=True,
                ignore_axes=None, allowed_channels=None, p_per_channel=0.5,
            ),
            apply_probability=0.25,
        ),
        RemoveLabelTansform(-1, 0),
    ]
    if deep_supervision_scales is not None:
        transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
    return ComposeTransforms(transforms)


class _SynthSegTrainLoader:
    """
    Prefetch wrapper: background thread calls BrainGenerator once and queues
    _N_CROPS individual patches.  Main thread consumes one patch per training step.

    Shared bg_lock serialises BrainGenerator across all threads (TF is not thread-safe).
    TF releases the GIL during model.predict → overlaps with PyTorch GPU compute.
    """

    def __init__(self, brain_generator, patch_size: tuple, bg_lock: threading.Lock) -> None:
        self._bg = brain_generator
        self._patch_size = patch_size
        self._lock = bg_lock
        self._q: queue.Queue = queue.Queue(maxsize=_N_CROPS * 4)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while True:
            with self._lock:
                batches = generate_synthseg_batch(self._bg, self._patch_size, n_crops=_N_CROPS)
            for b in batches:
                self._q.put(b)

    def __next__(self) -> dict:
        return self._q.get()

    def __iter__(self):
        return self

    def stop(self) -> None:
        pass


class _RealT1wValLoader:
    """
    Random-crop loader over preprocessed real T1w volumes.

    Used for SynthSeg validation — no synthesis, so model selection is unbiased.
    Works with any nnUNet dataset via the standard dataset.identifiers /
    dataset.load_case() API.
    """

    def __init__(self, dataset, patch_size: tuple) -> None:
        self._dataset = dataset
        self._keys = list(dataset.identifiers)
        self._patch_size = tuple(patch_size)
        self._rng = np.random.default_rng()

    def _random_crop(self, data: np.ndarray, seg: np.ndarray):
        cd, ch, cw = self._patch_size
        _, D, H, W = data.shape
        d0 = int(self._rng.integers(0, max(1, D - cd + 1)))
        h0 = int(self._rng.integers(0, max(1, H - ch + 1)))
        w0 = int(self._rng.integers(0, max(1, W - cw + 1)))
        return (
            data[:, d0:d0+cd, h0:h0+ch, w0:w0+cw],
            seg[:, d0:d0+cd, h0:h0+ch, w0:w0+cw],
        )

    def __next__(self) -> dict:
        key = str(self._rng.choice(self._keys))
        data_lazy, seg_lazy, *_ = self._dataset.load_case(key)
        data = np.array(data_lazy).astype(np.float32)
        seg = np.array(seg_lazy).astype(np.int16)
        data_c, seg_c = self._random_crop(data, seg)
        return {
            "data":   torch.from_numpy(data_c).float().unsqueeze(0),
            "target": torch.from_numpy(seg_c).to(torch.int16).unsqueeze(0),
            "keys":   [key],
        }

    def generate_train_batch(self) -> dict:
        return self.__next__()

    def __iter__(self):
        return self

    def stop(self) -> None:
        pass


class nnUNetTrainerSynthSeg(nnUNetTrainerFast):
    """
    SynthSeg synthesis method base.  Works with any nnUNet dataset.

    Subclasses must provide _labels_dir (Path to fold-specific BrainGenerator labels)
    and set synthseg_mode ("A" or "B").  Optionally override _log_wandb_images().
    """

    synthseg_mode: str = "A"

    @property
    @abstractmethod
    def _labels_dir(self) -> Path: ...

    # ── Data loaders ──────────────────────────────────────────────────────────

    def get_dataloaders(self):
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
        self._deep_supervision_scales = deep_supervision_scales

        self._train_transforms = _build_v26_fast_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            ignore_label=self.label_manager.ignore_label,
        )
        self._val_transforms = _build_synthseg_val_transforms(
            deep_supervision_scales, ignore_label=self.label_manager.ignore_label,
        )

        if not self._labels_dir.exists():
            raise FileNotFoundError(
                f"SynthSeg labels directory not found: {self._labels_dir}\n"
                "Run the splits creation script first."
            )

        # BrainGenerator (nibabel) uses (x,y,z); nnUNet (SimpleITK) uses (z,y,x).
        D, H, W = self._patch_size_cfg
        bg_target_shape = (W, H, D)
        self._brain_generator = build_brain_generator(
            self._labels_dir,
            mode=self.synthseg_mode,
            target_shape=bg_target_shape,
        )

        bg_lock = threading.Lock()
        dl_train = _SynthSegTrainLoader(self._brain_generator, self._patch_size_cfg, bg_lock)

        _, dataset_val = self.get_tr_and_val_datasets()
        dl_val = _RealT1wValLoader(dataset_val, self._patch_size_cfg)

        return dl_train, dl_val

    # ── train_step ────────────────────────────────────────────────────────────

    def train_step(self, batch: dict) -> dict:
        data_cpu = batch["data"][0].float()
        seg_cpu = batch["target"][0]

        # GPU spatial augmentation: random rotation + scaling (~9ms).
        data_gpu = data_cpu.unsqueeze(0).to(self.device, non_blocking=True)
        seg_gpu = seg_cpu.float().unsqueeze(0).to(self.device, non_blocking=True)
        data_gpu, seg_gpu = gpu_spatial_augment(data_gpu, seg_gpu, p_rotation=0.2, p_scaling=0.2)
        data_cpu = data_gpu.squeeze(0).cpu()
        seg_cpu = seg_gpu.squeeze(0).cpu().to(torch.int16)

        t = self._train_transforms(image=data_cpu, segmentation=seg_cpu)
        data_aug = t["image"].unsqueeze(0).to(self.device, non_blocking=True)
        seg_raw = t["segmentation"]
        if isinstance(seg_raw, (list, tuple)):
            seg_aug = [s.unsqueeze(0).to(self.device, non_blocking=True) for s in seg_raw]
        else:
            seg_aug = seg_raw.unsqueeze(0).to(self.device, non_blocking=True)

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
        data_cpu = batch["data"][0].float()
        seg_cpu = batch["target"][0]

        t = self._val_transforms(image=data_cpu, segmentation=seg_cpu)
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
        target_primary = seg_aug[0] if isinstance(seg_aug, list) else seg_aug

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
