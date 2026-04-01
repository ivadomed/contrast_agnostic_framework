from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


_SHARED_RNG_COUNTER = 0


def _next_shared_seed() -> int:
    global _SHARED_RNG_COUNTER
    _SHARED_RNG_COUNTER += 1
    seed = (int(torch.initial_seed()) + _SHARED_RNG_COUNTER) % (2**63 - 1)
    if dist.is_available() and dist.is_initialized():
        seed_tensor = torch.tensor([seed], dtype=torch.long)
        dist.broadcast(seed_tensor, src=0)
        seed = int(seed_tensor.item())
    return seed


def _shared_cpu_generator() -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_next_shared_seed())
    return generator


def _shared_rand(shape: tuple[int, ...], device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    dtype = dtype or torch.float32
    if not (dist.is_available() and dist.is_initialized()):
        return torch.rand(shape, device=device, dtype=dtype)
    rand_cpu = torch.rand(shape, generator=_shared_cpu_generator(), device="cpu", dtype=dtype)
    return rand_cpu.to(device=device, dtype=dtype)


def _shared_randint(low: int, high: int, size: tuple[int, ...], device: torch.device) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return torch.randint(low=low, high=high, size=size, device=device)
    randint_cpu = torch.randint(low=low, high=high, size=size, generator=_shared_cpu_generator(), device="cpu")
    return randint_cpu.to(device=device)


def _shared_randn_like(x: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return torch.randn_like(x)
    noise_cpu = torch.randn(x.shape, generator=_shared_cpu_generator(), device="cpu", dtype=x.dtype)
    return noise_cpu.to(device=x.device, dtype=x.dtype)


def _compute_quantile_centroids(sampled: torch.Tensor, q_probs: torch.Tensor) -> torch.Tensor:
    return torch.quantile(sampled, q_probs, dim=1).transpose(0, 1)


if hasattr(torch, "_dynamo"):
    _compute_quantile_centroids = torch._dynamo.disable(_compute_quantile_centroids)


def _batched_interp1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Vectorized 1D linear interpolation for batched inputs.

    Args:
        x: Query values shaped as (N, V).
        xp: Sorted x breakpoints shaped as (N, M).
        fp: y values at breakpoints shaped as (N, M).
    """
    if x.ndim != 2 or xp.ndim != 2 or fp.ndim != 2:
        raise ValueError("Expected x, xp, and fp to be 2D tensors.")
    if xp.shape != fp.shape:
        raise ValueError("xp and fp must have identical shapes.")
    if x.shape[0] != xp.shape[0]:
        raise ValueError("Batch dimension mismatch between x and xp/fp.")

    eps = torch.finfo(x.dtype).eps
    x_clamped = torch.maximum(x, xp[:, :1])
    x_clamped = torch.minimum(x_clamped, xp[:, -1:])

    idx = torch.searchsorted(xp, x_clamped, right=True)
    idx = idx.clamp(min=1, max=xp.shape[1] - 1)

    x0 = torch.gather(xp, 1, idx - 1)
    x1 = torch.gather(xp, 1, idx)
    y0 = torch.gather(fp, 1, idx - 1)
    y1 = torch.gather(fp, 1, idx)

    t = (x_clamped - x0) / (x1 - x0).clamp_min(eps)
    return y0 + t * (y1 - y0)


class RandomBezierIntensityWarp(nn.Module):
    """Apply per-sample cubic Bezier intensity warping on normalized volumes."""

    def __init__(self, p: float = 1.0) -> None:
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected 5D tensor (B, C, D, H, W), got shape {tuple(x.shape)}")
        if self.p <= 0.0:
            return x

        b = x.shape[0]
        apply_mask = (_shared_rand((b,), device=x.device) < self.p).view(b, 1, 1, 1, 1)
        if not bool(apply_mask.any()):
            return x

        p1 = _shared_rand((b, 1, 1, 1, 1), device=x.device, dtype=x.dtype)
        p2 = _shared_rand((b, 1, 1, 1, 1), device=x.device, dtype=x.dtype)

        x_clamped = x.clamp(0.0, 1.0)
        one_minus = 1.0 - x_clamped
        warped = (
            3.0 * (one_minus ** 2) * x_clamped * p1
            + 3.0 * one_minus * (x_clamped ** 2) * p2
            + (x_clamped ** 3)
        )
        warped = warped.clamp(0.0, 1.0)
        return torch.where(apply_mask, warped, x_clamped)


class RandomAnisotropicDegradation3D(nn.Module):
    """Simulate thick-slice artifacts by anisotropic depth down/up-sampling."""

    def __init__(self, p: float = 0.5, min_factor: int = 4, max_factor: int = 8) -> None:
        super().__init__()
        if min_factor <= 0 or max_factor < min_factor:
            raise ValueError("Expected 0 < min_factor <= max_factor.")
        self.p = float(p)
        self.min_factor = int(min_factor)
        self.max_factor = int(max_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected 5D tensor (B, C, D, H, W), got shape {tuple(x.shape)}")
        if self.p <= 0.0:
            return x

        b, _, d, h, w = x.shape
        apply_mask = _shared_rand((b,), device=x.device) < self.p
        if not bool(apply_mask.any()):
            return x

        output = x.clone()
        for i in range(b):
            if not bool(apply_mask[i]):
                continue

            factor = int(_shared_randint(self.min_factor, self.max_factor + 1, (1,), device=x.device).item())
            d_low = max(1, d // factor)

            low = F.interpolate(
                x[i : i + 1],
                size=(d_low, h, w),
                mode="area",
            )
            restored = F.interpolate(
                low,
                size=(d, h, w),
                mode="trilinear",
                align_corners=False,
            )
            output[i : i + 1] = restored

        return output.clamp(0.0, 1.0)


class RandomGMMHistogramMatching(nn.Module):
    """Random multi-peak GMM histogram matching for normalized 3D volumes."""

    def __init__(
        self,
        p: float = 1.0,
        num_quantiles: int = 100,
        sample_size: int = 100000,
        num_bins: int = 100,
        min_peaks: int = 3,
        max_peaks: int = 6,
    ) -> None:
        super().__init__()
        if num_quantiles < 2:
            raise ValueError("num_quantiles must be >= 2.")
        if num_bins < 2:
            raise ValueError("num_bins must be >= 2.")
        if sample_size <= 0:
            raise ValueError("sample_size must be positive.")
        if min_peaks <= 0 or max_peaks < min_peaks:
            raise ValueError("Expected 0 < min_peaks <= max_peaks.")

        self.p = float(p)
        self.num_quantiles = int(num_quantiles)
        self.sample_size = int(sample_size)
        self.num_bins = int(num_bins)
        self.min_peaks = int(min_peaks)
        self.max_peaks = int(max_peaks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected 5D tensor (B, C, D, H, W), got shape {tuple(x.shape)}")
        if self.p <= 0.0:
            return x.clamp(0.0, 1.0)

        b = x.shape[0]
        apply_mask = (_shared_rand((b,), device=x.device) < self.p).view(b, 1, 1, 1, 1)
        if not bool(apply_mask.any()):
            return x.clamp(0.0, 1.0)

        x_clamped = x.clamp(0.0, 1.0)
        bc = x_clamped.shape[0] * x_clamped.shape[1]
        flat = x_clamped.reshape(bc, -1)
        total_voxels = flat.shape[1]

        # Preserve black background: only sample from non-black pixels (threshold=1e-4)
        background_threshold = 1e-4
        is_tissue = flat > background_threshold
        
        # Create source quantiles by sampling from tissue pixels only
        q_probs = torch.linspace(0.0, 1.0, self.num_quantiles, device=x.device, dtype=x.dtype)
        source_quantiles_list = []
        
        for bc_idx in range(bc):
            tissue_pixels = flat[bc_idx, is_tissue[bc_idx]]
            if tissue_pixels.numel() < 2:
                # If no tissue pixels, use the full histogram
                tissue_pixels = flat[bc_idx]
            
            source_quantiles_list.append(torch.quantile(tissue_pixels, q_probs))
        
        source_quantiles = torch.stack(source_quantiles_list, dim=0)  # (bc, num_quantiles)

        bin_centers = torch.linspace(0.0, 1.0, self.num_bins, device=x.device, dtype=x.dtype)
        peak_count = _shared_randint(
            low=self.min_peaks,
            high=self.max_peaks + 1,
            size=(bc, 1),
            device=x.device,
        )
        peak_ids = torch.arange(self.max_peaks, device=x.device).view(1, -1)
        active = (peak_ids < peak_count).to(x.dtype)

        means = _shared_rand((bc, self.max_peaks), device=x.device, dtype=x.dtype)
        sigmas = 0.02 + 0.08 * _shared_rand((bc, self.max_peaks), device=x.device, dtype=x.dtype)
        weights = (0.1 + 0.9 * _shared_rand((bc, self.max_peaks), device=x.device, dtype=x.dtype)) * active
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(x.dtype).eps)

        gauss_input = (bin_centers.view(1, 1, self.num_bins) - means.unsqueeze(-1)) / sigmas.unsqueeze(-1)
        gaussians = torch.exp(-0.5 * gauss_input.square()) / (
            sigmas.unsqueeze(-1) * (2.0 * torch.pi) ** 0.5
        )
        gmm_pdf = (weights.unsqueeze(-1) * gaussians).sum(dim=1)
        gmm_pdf = gmm_pdf / gmm_pdf.sum(dim=1, keepdim=True).clamp_min(torch.finfo(x.dtype).eps)

        target_cdf = torch.cumsum(gmm_pdf, dim=1)
        target_cdf = target_cdf / target_cdf[:, -1:].clamp_min(torch.finfo(x.dtype).eps)

        target_quantiles = _batched_interp1d(
            q_probs.view(1, -1).expand(bc, -1),
            target_cdf,
            bin_centers.view(1, -1).expand(bc, -1),
        )

        mapped_flat = _batched_interp1d(flat, source_quantiles, target_quantiles)
        
        # Force background pixels back to 0 to preserve black background
        mapped_flat[~is_tissue] = 0.0
        
        mapped = mapped_flat.view_as(x_clamped).clamp(0.0, 1.0)
        return torch.where(apply_mask, mapped, x_clamped)


class RandomSoftQuantileShuffling(nn.Module):
    """Non-monotonic soft quantile shuffling for normalized 3D volumes."""

    def __init__(
        self,
        p: float = 1.0,
        num_centroids: int = 5,
        sample_size: int = 100000,
        temperature: float = 0.05,
        noise_std: float = 0.02,
        preserve_background: bool = True,
        background_threshold: float = 1e-4,
    ) -> None:
        super().__init__()
        if num_centroids < 2:
            raise ValueError("num_centroids must be >= 2.")
        if sample_size <= 0:
            raise ValueError("sample_size must be positive.")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0.")
        if noise_std < 0.0:
            raise ValueError("noise_std must be >= 0.")

        self.p = float(p)
        self.num_centroids = int(num_centroids)
        self.sample_size = int(sample_size)
        self.temperature = float(temperature)
        self.noise_std = float(noise_std)
        self.preserve_background = bool(preserve_background)
        self.background_threshold = float(background_threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected 5D tensor (B, C, D, H, W), got shape {tuple(x.shape)}")
        if self.p <= 0.0:
            return x.clamp(0.0, 1.0)

        b = x.shape[0]
        apply_mask = (_shared_rand((b,), device=x.device) < self.p).view(b, 1, 1, 1, 1)
        if not bool(apply_mask.any()):
            return x.clamp(0.0, 1.0)

        x_clamped = x.clamp(0.0, 1.0)
        bc = x_clamped.shape[0] * x_clamped.shape[1]
        flat = x_clamped.reshape(bc, -1)
        total_voxels = flat.shape[1]

        sample_size = min(self.sample_size, total_voxels)
        sample_idx = _shared_randint(
            low=0,
            high=total_voxels,
            size=(sample_size,),
            device=x.device,
        )
        sampled = flat[:, sample_idx]

        q_start = 0.5 / float(self.num_centroids)
        q_end = 1.0 - q_start
        q_probs = torch.linspace(
            q_start,
            q_end,
            self.num_centroids,
            device=x.device,
            dtype=x.dtype,
        )
        # torch.quantile can fail under torch.compile/TorchDynamo for this dynamic path.
        # Run this small step in eager mode, then continue with vectorized tensor ops.
        centroids = _compute_quantile_centroids(sampled, q_probs)

        with torch.autocast(device_type=x.device.type, enabled=False):
            flat_fp32 = flat.to(torch.float32)
            centroids_fp32 = centroids.to(torch.float32)
            diffs = flat_fp32.unsqueeze(-1) - centroids_fp32.unsqueeze(1)
            logits = -diffs.square() / float(self.temperature)
            weights = torch.softmax(logits, dim=-1)
        weights = weights.to(x.dtype)

        targets = _shared_rand((bc, self.num_centroids), device=x.device, dtype=x.dtype)
        mapped_flat = torch.sum(weights * targets.unsqueeze(1), dim=-1)

        if self.noise_std > 0.0:
            mapped_flat = mapped_flat + self.noise_std * _shared_randn_like(mapped_flat)

        if self.preserve_background:
            bg_mask = flat <= self.background_threshold
            mapped_flat = torch.where(bg_mask, torch.zeros_like(mapped_flat), mapped_flat)

        mapped = mapped_flat.view_as(x_clamped).clamp(0.0, 1.0)
        return torch.where(apply_mask, mapped, x_clamped)


class RandomSpatialSoftQuantile(nn.Module):
    """Spatially-varying soft quantile shuffling for normalized 3D volumes."""

    def __init__(
        self,
        p: float = 1.0,
        num_centroids: int = 5,
        sample_size: int = 100000,
        temperature: float = 0.05,
        noise_std: float = 0.02,
        coarse_size: tuple[int, int, int] = (3, 3, 3),
        preserve_background: bool = True,
        background_threshold: float = 1e-4,
    ) -> None:
        super().__init__()
        if num_centroids < 2:
            raise ValueError("num_centroids must be >= 2.")
        if sample_size <= 0:
            raise ValueError("sample_size must be positive.")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0.")
        if noise_std < 0.0:
            raise ValueError("noise_std must be >= 0.")
        if len(coarse_size) != 3 or any(int(v) <= 0 for v in coarse_size):
            raise ValueError("coarse_size must be a tuple of three positive integers.")

        self.p = float(p)
        self.num_centroids = int(num_centroids)
        self.sample_size = int(sample_size)
        self.temperature = float(temperature)
        self.noise_std = float(noise_std)
        self.coarse_size = tuple(int(v) for v in coarse_size)
        self.preserve_background = bool(preserve_background)
        self.background_threshold = float(background_threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected 5D tensor (B, C, D, H, W), got shape {tuple(x.shape)}")
        if self.p <= 0.0:
            return x.clamp(0.0, 1.0)

        b = x.shape[0]
        apply_mask = (_shared_rand((b,), device=x.device) < self.p).view(b, 1, 1, 1, 1)
        if not bool(apply_mask.any()):
            return x.clamp(0.0, 1.0)

        x_clamped = x.clamp(0.0, 1.0)
        _, _, d, h, w = x_clamped.shape
        bc = x_clamped.shape[0] * x_clamped.shape[1]
        flat = x_clamped.reshape(bc, -1)
        total_voxels = flat.shape[1]

        sample_size = min(self.sample_size, total_voxels)
        sample_idx = _shared_randint(
            low=0,
            high=total_voxels,
            size=(sample_size,),
            device=x.device,
        )
        sampled = flat[:, sample_idx]

        q_start = 0.5 / float(self.num_centroids)
        q_end = 1.0 - q_start
        q_probs = torch.linspace(
            q_start,
            q_end,
            self.num_centroids,
            device=x.device,
            dtype=x.dtype,
        )
        centroids = _compute_quantile_centroids(sampled, q_probs)

        with torch.autocast(device_type=x.device.type, enabled=False):
            flat_fp32 = flat.to(torch.float32)
            centroids_fp32 = centroids.to(torch.float32)
            diffs = flat_fp32.unsqueeze(-1) - centroids_fp32.unsqueeze(1)
            logits = -diffs.square() / float(self.temperature)
            weights = torch.softmax(logits, dim=-1)
        weights = weights.to(x.dtype)
        weights_spatial = weights.transpose(1, 2).reshape(bc, self.num_centroids, d, h, w)

        coarse_targets = _shared_rand(
            (bc, self.num_centroids, self.coarse_size[0], self.coarse_size[1], self.coarse_size[2]),
            device=x.device,
            dtype=x.dtype,
        )
        spatial_targets = F.interpolate(
            coarse_targets,
            size=(d, h, w),
            mode="trilinear",
            align_corners=True,
        )

        mapped = torch.sum(weights_spatial * spatial_targets, dim=1, keepdim=True)
        mapped_flat = mapped.reshape(bc, -1)

        if self.noise_std > 0.0:
            mapped_flat = mapped_flat + self.noise_std * _shared_randn_like(mapped_flat)

        if self.preserve_background:
            bg_mask = flat <= self.background_threshold
            mapped_flat = torch.where(bg_mask, torch.zeros_like(mapped_flat), mapped_flat)

        mapped = mapped_flat.view_as(x_clamped).clamp(0.0, 1.0)
        return torch.where(apply_mask, mapped, x_clamped)