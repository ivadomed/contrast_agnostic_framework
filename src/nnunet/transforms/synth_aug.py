"""
GPU/CPU synthesis utilities for V26_6SignedAlphaTargetGenerator.

Two entry points:
  synthesize_volume(data, seg, ...)   — full pipeline (step 1 = min-max norm included)
  synthesize_patch(image_01, seg, ...) — steps 2-5 only (assumes already in [0,1])

synthesize_patch is used by background CPU workers: the caller normalises the full
volume first, then crops a patch, then calls synthesize_patch on that patch.
This keeps the full-volume min-max values (so CSF is not dark) while synthesising
only the patch (3.4× smaller = much faster).

Normalization pipeline matches generate_synthetic_guidance.py exactly:
  1. Full-volume min-max → [0, 1]  (ScaleIntensityd)
  2. V26_6(image_01, hist_module)  — no labels=
  3. Native-resolution Gaussian blur  (sigma ∈ {0,0.3,0.5,0.8})
  4. Zero voxels where image_01 < 0.01
  5. Re-z-score within brain (seg > 0)
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

_BLUR_SIGMAS    = [0.0, 0.0, 0.0, 0.3, 0.5, 0.8]
_DARK_THRESHOLD = 0.01


@torch.no_grad()
def synthesize_volume(
    data: torch.Tensor,
    seg: torch.Tensor,
    generator,
    hist_module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Full synthesis pipeline (steps 1-5).

    Parameters
    ----------
    data : (1, C, D, H, W) z-score normalized T1w on any device
    seg  : (1, 1, D, H, W) int16 GT labels on same device

    Returns
    -------
    (synth_z, synth_01) — both (1, C, D, H, W)
    """
    v_min = data.min()
    v_max = data.max()
    image_01 = (data - v_min) / (v_max - v_min + 1e-7)
    image_01 = image_01.clamp(0.0, 1.0)
    return _synth_from_01(image_01, seg, generator, hist_module)


@torch.no_grad()
def synthesize_patch(
    image_01: torch.Tensor,
    seg: torch.Tensor,
    generator,
    hist_module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Synthesis steps 2-5 — assumes image_01 is already in [0, 1] from FULL-VOLUME min-max.

    Called by CPU background workers after normalising the full volume and cropping
    a patch.  Works on CPU or GPU tensors depending on where inputs live.

    Parameters
    ----------
    image_01 : (1, C, D, H, W) already in [0, 1]
    seg      : (1, 1, D, H, W) int-type labels

    Returns
    -------
    (synth_z, synth_01) — both (1, C, D, H, W) on same device as input
    """
    return _synth_from_01(image_01, seg, generator, hist_module)


def _synth_from_01(image_01, seg, generator, hist_module):
    """Internal: synthesis steps 2-5 given a [0,1]-normalised volume."""
    eps = 1e-7
    brain_mask = (seg[0, 0] > 0)

    _, synth_01, _ = generator(image_01, hist_module)

    sigma = random.choice(_BLUR_SIGMAS)
    if sigma > 0.0:
        k_r = max(1, int(3.0 * sigma + 0.5))
        k1d = torch.arange(-k_r, k_r + 1, dtype=synth_01.dtype, device=synth_01.device)
        k1d = torch.exp(-0.5 * (k1d / sigma) ** 2)
        k1d = k1d / k1d.sum()
        pad = len(k1d) // 2
        B, C, D, H, W = synth_01.shape
        y = synth_01.view(B * C, 1, D, H, W)
        y = F.conv3d(y, k1d.view(1, 1, -1, 1, 1), padding=(pad, 0, 0))
        y = F.conv3d(y, k1d.view(1, 1, 1, -1, 1), padding=(0, pad, 0))
        y = F.conv3d(y, k1d.view(1, 1, 1, 1, -1), padding=(0, 0, pad))
        synth_01 = y.view(B, C, D, H, W)
    synth_01 = synth_01.clamp(0.0, 1.0)

    synth_01 = torch.where(
        image_01 < _DARK_THRESHOLD,
        torch.zeros_like(synth_01),
        synth_01,
    )

    synth_brain = synth_01[0, 0][brain_mask]
    synth_z = torch.zeros_like(synth_01)
    if synth_brain.numel() > 0:
        synth_z[0, 0][brain_mask] = (
            (synth_brain - synth_brain.mean()) / (synth_brain.std() + eps)
        )

    return synth_z, synth_01


def random_crop_pair(
    data: torch.Tensor,
    seg: torch.Tensor,
    crop_size: tuple[int, int, int],
    n_crops: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly crop n_crops patches from a full volume.

    Parameters
    ----------
    data     : (1, C, D, H, W)
    seg      : (1, 1, D, H, W)
    crop_size: (cd, ch, cw)
    n_crops  : number of patches

    Returns
    -------
    patches     : (n_crops, C, cd, ch, cw)
    patches_seg : (n_crops, 1, cd, ch, cw)
    """
    _, C, D, H, W = data.shape
    cd, ch, cw = crop_size

    data_crops = []
    seg_crops  = []
    for _ in range(n_crops):
        d0 = torch.randint(0, max(1, D - cd + 1), (1,)).item()
        h0 = torch.randint(0, max(1, H - ch + 1), (1,)).item()
        w0 = torch.randint(0, max(1, W - cw + 1), (1,)).item()

        d1 = min(d0 + cd, D); d0 = d1 - cd
        h1 = min(h0 + ch, H); h0 = h1 - ch
        w1 = min(w0 + cw, W); w0 = w1 - cw

        data_crops.append(data[0, :, d0:d1, h0:h1, w0:w1])
        seg_crops.append(seg[0, :, d0:d1, h0:h1, w0:w1])

    return torch.stack(data_crops), torch.stack(seg_crops)


def center_crop_pair(
    data: torch.Tensor,
    seg: torch.Tensor,
    crop_size: tuple[int, int, int],
    n_crops: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Center crop — used for validation and image logging."""
    _, C, D, H, W = data.shape
    cd, ch, cw = crop_size

    d0 = max(0, (D - cd) // 2); d1 = min(d0 + cd, D); d0 = d1 - cd
    h0 = max(0, (H - ch) // 2); h1 = min(h0 + ch, H); h0 = h1 - ch
    w0 = max(0, (W - cw) // 2); w1 = min(w0 + cw, W); w0 = w1 - cw

    patch     = data[0, :, d0:d1, h0:h1, w0:w1].unsqueeze(0).expand(n_crops, -1, -1, -1, -1)
    patch_seg = seg[0, :, d0:d1, h0:h1, w0:w1].unsqueeze(0).expand(n_crops, -1, -1, -1, -1)
    return patch.contiguous(), patch_seg.contiguous()
