from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.synthesis.intensity_ops import (
    RandomBezierIntensityWarp,
    RandomGMMHistogramMatching,
    RandomSoftQuantileShuffling,
)
from src.synthesis.noise_ops import generate_fractal_noise_3d


class BaseGuidancePerturber(nn.Module):
    """Strategy interface for optional guidance perturbations."""

    def forward(self, guidance_map: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class IdentityGuidancePerturber(BaseGuidancePerturber):
    def forward(self, guidance_map: torch.Tensor) -> torch.Tensor:
        return guidance_map


class ProceduralNoiseGuidancePerturber(BaseGuidancePerturber):
    def __init__(self, noise_strength: float = 0.2, noise_dtype: str = "float16"):
        super().__init__()
        self.noise_strength = float(noise_strength)
        self.noise_dtype = noise_dtype

    def forward(self, guidance_map: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            procedural_noise = generate_fractal_noise_3d(
                guidance_map.detach(),
                noise_dtype=self.noise_dtype,
            )
        return (guidance_map + self.noise_strength * procedural_noise).clamp(0.0, 1.0)


class BezierWarpGuidancePerturber(BaseGuidancePerturber):
    def __init__(self, p: float = 1.0):
        super().__init__()
        self._op = RandomBezierIntensityWarp(p=p)

    def forward(self, guidance_map: torch.Tensor) -> torch.Tensor:
        return self._op(guidance_map)


class GMMMatchingGuidancePerturber(BaseGuidancePerturber):
    def __init__(self, p: float = 1.0):
        super().__init__()
        self._op = RandomGMMHistogramMatching(p=p)

    def forward(self, guidance_map: torch.Tensor) -> torch.Tensor:
        return self._op(guidance_map)


class SoftQuantileShuffleGuidancePerturber(BaseGuidancePerturber):
    def __init__(self, p: float = 1.0):
        super().__init__()
        self._op = RandomSoftQuantileShuffling(p=p)

    def forward(self, guidance_map: torch.Tensor) -> torch.Tensor:
        return self._op(guidance_map)


class HeavyGaussianGuidancePerturber(BaseGuidancePerturber):
    """Heavy low-pass guidance perturbation using separable 1D depthwise 3D convolutions."""

    def __init__(
        self,
        sigma: float = 3.0,
        kernel_size: int = 21,
        background_threshold: float = 0.01,
    ):
        super().__init__()
        if sigma <= 0.0:
            raise ValueError("sigma must be positive.")
        if kernel_size < 3:
            raise ValueError("kernel_size must be >= 3.")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")

        self.sigma = float(sigma)
        self.kernel_size = int(kernel_size)
        self.background_threshold = float(background_threshold)

    def _kernel_1d(self, x: torch.Tensor) -> torch.Tensor:
        coords = torch.arange(self.kernel_size, dtype=x.dtype, device=x.device)
        coords = coords - (self.kernel_size - 1) / 2.0
        g1d = torch.exp(-(coords ** 2) / (2.0 * self.sigma ** 2))
        return g1d / g1d.sum().clamp_min(torch.finfo(g1d.dtype).eps)

    def forward(self, guidance_map: torch.Tensor) -> torch.Tensor:
        if guidance_map.ndim != 5:
            raise ValueError("guidance_map must be a 5D tensor shaped as (B, C, D, H, W).")

        channels = guidance_map.shape[1]
        orig_dtype = guidance_map.dtype
        g1d = self._kernel_1d(guidance_map)
        k = g1d.shape[0]
        pad = k // 2

        k_d = g1d.view(1, 1, k, 1, 1).expand(channels, 1, k, 1, 1).contiguous()
        k_h = g1d.view(1, 1, 1, k, 1).expand(channels, 1, 1, k, 1).contiguous()
        k_w = g1d.view(1, 1, 1, 1, k).expand(channels, 1, 1, 1, k).contiguous()

        x32 = guidance_map.to(torch.float32)
        x32 = F.pad(x32, (0, 0, 0, 0, pad, pad), mode="replicate")
        y32 = F.conv3d(x32, k_d.to(torch.float32), groups=channels)
        y32 = F.pad(y32, (0, 0, pad, pad, 0, 0), mode="replicate")
        y32 = F.conv3d(y32, k_h.to(torch.float32), groups=channels)
        y32 = F.pad(y32, (pad, pad, 0, 0, 0, 0), mode="replicate")
        y32 = F.conv3d(y32, k_w.to(torch.float32), groups=channels)

        y = y32.clamp(0.0, 1.0).to(dtype=orig_dtype)
        y[guidance_map < self.background_threshold] = 0.0
        return y


class MildGaussianGuidancePerturber(BaseGuidancePerturber):
    """Mild low-pass guidance perturbation using separable 1D depthwise 3D convolutions."""

    def __init__(
        self,
        sigma: float = 1.0,
        kernel_size: int = 9,
        background_threshold: float = 0.02,
    ):
        super().__init__()
        if sigma <= 0.0:
            raise ValueError("sigma must be positive.")
        if kernel_size < 3:
            raise ValueError("kernel_size must be >= 3.")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")

        self.sigma = float(sigma)
        self.kernel_size = int(kernel_size)
        self.background_threshold = float(background_threshold)

    def _kernel_1d(self, x: torch.Tensor) -> torch.Tensor:
        coords = torch.arange(self.kernel_size, dtype=x.dtype, device=x.device)
        coords = coords - (self.kernel_size - 1) / 2.0
        g1d = torch.exp(-(coords ** 2) / (2.0 * self.sigma ** 2))
        return g1d / g1d.sum().clamp_min(torch.finfo(g1d.dtype).eps)

    def forward(self, guidance_map: torch.Tensor) -> torch.Tensor:
        if guidance_map.ndim != 5:
            raise ValueError("guidance_map must be a 5D tensor shaped as (B, C, D, H, W).")

        channels = guidance_map.shape[1]
        orig_dtype = guidance_map.dtype
        g1d = self._kernel_1d(guidance_map)
        k = g1d.shape[0]
        pad = k // 2

        k_d = g1d.view(1, 1, k, 1, 1).expand(channels, 1, k, 1, 1).contiguous()
        k_h = g1d.view(1, 1, 1, k, 1).expand(channels, 1, 1, k, 1).contiguous()
        k_w = g1d.view(1, 1, 1, 1, k).expand(channels, 1, 1, 1, k).contiguous()

        x32 = guidance_map.to(torch.float32)
        x32 = F.pad(x32, (0, 0, 0, 0, pad, pad), mode="replicate")
        y32 = F.conv3d(x32, k_d.to(torch.float32), groups=channels)
        y32 = F.pad(y32, (0, 0, pad, pad, 0, 0), mode="replicate")
        y32 = F.conv3d(y32, k_h.to(torch.float32), groups=channels)
        y32 = F.pad(y32, (pad, pad, 0, 0, 0, 0), mode="replicate")
        y32 = F.conv3d(y32, k_w.to(torch.float32), groups=channels)

        y = y32.clamp(0.0, 1.0).to(dtype=orig_dtype)
        y[guidance_map < self.background_threshold] = 0.0
        return y
