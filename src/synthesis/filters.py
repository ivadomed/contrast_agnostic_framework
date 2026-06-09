from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.synthesis.histogram_ops import generate_non_monotonic_grid_targets


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


class DifferentiableLPCI3D(nn.Module):
	"""Differentiable Laplacian Pyramid Contrast Inversion for 3D MRI guidance."""

	def __init__(
		self,
		alpha_min: float = 0.8,
		alpha_max: float = 1.2,
		background_threshold: float = 0.01,
	) -> None:
		super().__init__()
		self.alpha_min = float(alpha_min)
		self.alpha_max = float(alpha_max)
		self.background_threshold = float(background_threshold)

	@staticmethod
	def _kernel_size_for_sigma(sigma: float) -> int:
		kernel_size = int(max(3, round(float(sigma) * 6.0)))
		if kernel_size % 2 == 0:
			kernel_size += 1
		return kernel_size

	def _separable_gaussian_blur(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
		if sigma <= 0.0:
			return x

		channels = x.shape[1]
		kernel_size = self._kernel_size_for_sigma(sigma)
		coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - (kernel_size - 1) / 2.0
		g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
		g1d = g1d / g1d.sum().clamp_min(torch.finfo(g1d.dtype).eps)

		padding = kernel_size // 2
		k_d = g1d.view(1, 1, kernel_size, 1, 1).expand(channels, 1, kernel_size, 1, 1).contiguous()
		k_h = g1d.view(1, 1, 1, kernel_size, 1).expand(channels, 1, 1, kernel_size, 1).contiguous()
		k_w = g1d.view(1, 1, 1, 1, kernel_size).expand(channels, 1, 1, 1, kernel_size).contiguous()

		smoothed = F.conv3d(x, k_d, padding=(padding, 0, 0), groups=channels)
		smoothed = F.conv3d(smoothed, k_h, padding=(0, padding, 0), groups=channels)
		smoothed = F.conv3d(smoothed, k_w, padding=(0, 0, padding), groups=channels)
		return smoothed

	def forward(
		self,
		x: torch.Tensor,
		hist_module,
		num_chunks: int,
		dark_threshold: float,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		if x.ndim != 5:
			raise ValueError(f"Expected a 5D tensor (B, C, D, H, W), got shape {tuple(x.shape)}")

		b, _, d, h, w = x.shape
		max_dim = float(max(d, h, w))
		sigma_1 = max(max_dim / 32.0, 0.5)
		sigma_2 = max(max_dim / 16.0, sigma_1 + 1e-3)

		g0 = x
		g1 = self._separable_gaussian_blur(g0, sigma=sigma_1)
		g2 = self._separable_gaussian_blur(g1, sigma=sigma_2)

		l0 = g0 - g1
		l1 = g1 - g2
		l2 = g2.clamp(0.0, 1.0)
		l2_fp32 = l2.float()

		_, perms, l2_prime = generate_non_monotonic_grid_targets(
			input_images=l2_fp32,
			num_chunks=int(num_chunks),
			dark_threshold=float(dark_threshold),
			hist_module=hist_module,
			background_threshold=self.background_threshold,
		)
		l2_prime = l2_prime.to(dtype=x.dtype)

		if self.alpha_max <= self.alpha_min:
			alpha = torch.full((b, 1, 1, 1, 1), self.alpha_min, device=x.device, dtype=x.dtype)
		else:
			alpha = torch.empty((b, 1, 1, 1, 1), device=x.device, dtype=x.dtype).uniform_(self.alpha_min, self.alpha_max)

		l0_prime = alpha * l0
		x_synth = (l0_prime + l1 + l2_prime).clamp(0.0, 1.0)
		target_hist = hist_module(x_synth)
		return target_hist, perms, x_synth
