from __future__ import annotations

import torch
from torch import nn

from src.intensity_ops import (
    RandomBezierIntensityWarp,
    RandomGMMHistogramMatching,
    RandomSoftQuantileShuffling,
)
from src.noise_ops import generate_fractal_noise_3d


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
