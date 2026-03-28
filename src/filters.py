from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnatomicalUnsharpMask3D(nn.Module):
	"""Sharpen anatomical boundaries using separable 3D Gaussian unsharp masking."""

	def __init__(self, alpha: float = 2.0, sigma: float = 1.0) -> None:
		super().__init__()
		if sigma <= 0.0:
			raise ValueError("sigma must be positive.")
		self.alpha = float(alpha)
		self.sigma = float(sigma)

	def _kernel_1d(self, x: torch.Tensor) -> torch.Tensor:
		kernel_size = int(max(3, round(self.sigma * 6)))
		if kernel_size % 2 == 0:
			kernel_size += 1

		coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - (kernel_size - 1) / 2.0
		g1d = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
		return g1d / g1d.sum().clamp_min(torch.finfo(g1d.dtype).eps)

	def _separable_gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
		channels = x.shape[1]
		g1d = self._kernel_1d(x)
		k = g1d.shape[0]
		padding = k // 2

		k_d = g1d.view(1, 1, k, 1, 1).expand(channels, 1, k, 1, 1).contiguous()
		k_h = g1d.view(1, 1, 1, k, 1).expand(channels, 1, 1, k, 1).contiguous()
		k_w = g1d.view(1, 1, 1, 1, k).expand(channels, 1, 1, 1, k).contiguous()

		smoothed = F.conv3d(x, k_d, padding=(padding, 0, 0), groups=channels)
		smoothed = F.conv3d(smoothed, k_h, padding=(0, padding, 0), groups=channels)
		smoothed = F.conv3d(smoothed, k_w, padding=(0, 0, padding), groups=channels)
		return smoothed

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"Expected a 5D tensor (B, C, D, H, W), got shape {tuple(x.shape)}")

		x_blurred = self._separable_gaussian_blur(x)
		edges = x - x_blurred
		x_sharpened = x + self.alpha * edges
		return x_sharpened.clamp(0.0, 1.0)
