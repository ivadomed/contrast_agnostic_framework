from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RandomResolutionResampler3D(nn.Module):
    """Simulate different acquisition resolutions by downsampling then upsampling back.

    Covers isotropic and anisotropic cases (e.g. thick slices in one axis).
    Intended as post-processing on synthesized images so the segmenter is
    robust to varying voxel sizes.

    Args:
        zoom_range: (min, max) uniform zoom factor per axis. Values < 1
            downsample (simulate lower resolution); values > 1 upsample
            (simulate super-resolution input), both followed by resampling
            back to the original spatial size.
        anisotropy_prob: probability of sampling different zoom factors per
            axis (anisotropic). Otherwise all three axes share one factor.
        apply_prob: probability of applying the resampling at all per call.
    """

    def __init__(
        self,
        zoom_range: tuple[float, float] = (0.3, 1.5),
        anisotropy_prob: float = 0.5,
        apply_prob: float = 1.0,
    ):
        super().__init__()
        self.zoom_min = float(zoom_range[0])
        self.zoom_max = float(zoom_range[1])
        self.anisotropy_prob = float(anisotropy_prob)
        self.apply_prob = float(apply_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.apply_prob < 1.0 and torch.rand(1).item() > self.apply_prob:
            return x

        original_size = x.shape[2:]  # (D, H, W)

        span = self.zoom_max - self.zoom_min
        if torch.rand(1).item() < self.anisotropy_prob:
            zooms = [torch.rand(1).item() * span + self.zoom_min for _ in range(3)]
        else:
            z = torch.rand(1).item() * span + self.zoom_min
            zooms = [z, z, z]

        new_size = [max(1, round(s * zf)) for s, zf in zip(original_size, zooms)]

        if new_size == list(original_size):
            return x

        downsampled = F.interpolate(x, size=new_size, mode="trilinear", align_corners=False)
        return F.interpolate(downsampled, size=original_size, mode="trilinear", align_corners=False)
