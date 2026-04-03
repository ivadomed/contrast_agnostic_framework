from __future__ import annotations

from typing import Protocol

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from src.intensity_ops import _shared_rand as _shared_rand_intensity
from src.intensity_ops import _shared_randint as _shared_randint_intensity


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


def _shared_randperm(n: int, device: torch.device) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return torch.randperm(n, device=device)
    perm_cpu = torch.randperm(n, generator=_shared_cpu_generator(), device="cpu")
    return perm_cpu.to(device=device)


def _shared_rand(shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return torch.rand(shape, device=device, dtype=dtype)
    rand_cpu = torch.rand(shape, generator=_shared_cpu_generator(), device="cpu", dtype=dtype)
    return rand_cpu.to(device=device, dtype=dtype)


class HistogramModuleLike(Protocol):
    num_bins: int
    min_value: float
    max_value: float

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...


class BaseTargetGenerator(nn.Module):
    """Strategy interface for guidance and target histogram generation."""

    def forward(
        self,
        input_images: torch.Tensor,
        num_bins: int,
        num_chunks: int,
        dark_threshold: float,
        hist_module: HistogramModuleLike,
        return_guidance_map: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class LegacyChunkTargetGenerator(BaseTargetGenerator):
    """Original percentile chunk remapping used by v1-v7 and default fallbacks."""

    def forward(
        self,
        input_images: torch.Tensor,
        num_bins: int,
        num_chunks: int,
        dark_threshold: float,
        hist_module: HistogramModuleLike,
        return_guidance_map: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if input_images.ndim != 5:
            raise ValueError("input_images must be a 5D tensor shaped as (B, C, D, H, W).")

        b = input_images.shape[0]
        perms = [_shared_randperm(num_chunks, device=input_images.device) for _ in range(b)]
        perms_tensor = torch.stack(perms, dim=0)

        with torch.no_grad():
            guidance_map = _create_range_translation_guidance_map(
                input_image=input_images,
                perms=perms_tensor,
                num_chunks=num_chunks,
                dark_threshold=dark_threshold,
            )

        target_hist = hist_module(guidance_map)
        return target_hist, perms_tensor, guidance_map


class V8GridTargetGenerator(BaseTargetGenerator):
    """Spatially varying monotonic chunk mapping with trilinear local quantiles."""

    def __init__(self, grid_size: tuple[int, int, int] = (4, 4, 4)):
        super().__init__()
        self.grid_size = tuple(int(v) for v in grid_size)

    def forward(
        self,
        input_images: torch.Tensor,
        num_bins: int,
        num_chunks: int,
        dark_threshold: float,
        hist_module: HistogramModuleLike,
        return_guidance_map: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if input_images.ndim != 5:
            raise ValueError("input_images must be a 5D tensor shaped as (B, C, D, H, W).")

        b, c, d, h, w = input_images.shape
        gz, gy, gx = self.grid_size
        if gz <= 0 or gy <= 0 or gx <= 0:
            raise ValueError("grid_size values must be positive integers.")

        perms = [_shared_randperm(num_chunks, device=input_images.device) for _ in range(b)]
        perms_tensor = torch.stack(perms, dim=0)

        x_scalar = input_images.mean(dim=1, keepdim=True)

        pad_d = (gz - (d % gz)) % gz
        pad_h = (gy - (h % gy)) % gy
        pad_w = (gx - (w % gx)) % gx
        if pad_d or pad_h or pad_w:
            x_scalar_pad = F.pad(x_scalar, (0, pad_w, 0, pad_h, 0, pad_d), mode="replicate")
        else:
            x_scalar_pad = x_scalar

        d_pad, h_pad, w_pad = x_scalar_pad.shape[-3:]
        bd = d_pad // gz
        bh = h_pad // gy
        bw = w_pad // gx

        block_values = (
            x_scalar_pad.view(b, 1, gz, bd, gy, bh, gx, bw)
            .permute(0, 2, 4, 6, 1, 3, 5, 7)
            .reshape(b, gz * gy * gx, -1)
        )

        block_values = block_values.clone()
        block_values[block_values <= dark_threshold] = float("nan")

        q_probs = torch.linspace(0.0, 1.0, num_chunks + 1, device=input_images.device, dtype=torch.float32)
        local_edges = torch.nanquantile(block_values, q_probs, dim=-1).to(input_images.dtype).permute(1, 0, 2)

        local_edges = torch.nan_to_num(local_edges, nan=float(dark_threshold))
        local_edges[:, 0, :] = torch.clamp(local_edges[:, 0, :], max=dark_threshold)
        local_edges[:, -1, :] = torch.clamp(local_edges[:, -1, :], min=1.0)

        local_edges_grid = local_edges.view(b, num_chunks + 1, gz, gy, gx)
        dense_edges = F.interpolate(
            local_edges_grid,
            size=(d, h, w),
            mode="trilinear",
            align_corners=True,
        )
        dense_edges = torch.cummax(dense_edges, dim=1).values

        b_idx = torch.arange(b, device=input_images.device)[:, None]
        chunk_idx = torch.arange(num_chunks, device=input_images.device)[None, :].expand(b, num_chunks)
        inverse_perm = torch.empty_like(perms_tensor)
        inverse_perm[b_idx, perms_tensor] = chunk_idx

        x_vals = x_scalar.squeeze(1)
        bin_idx = torch.sum(x_vals.unsqueeze(1) > dense_edges, dim=1) - 1
        bin_idx = torch.clamp(bin_idx, 0, num_chunks - 1)

        source_lower = torch.gather(dense_edges, 1, bin_idx.unsqueeze(1)).squeeze(1)
        source_upper = torch.gather(dense_edges, 1, (bin_idx + 1).unsqueeze(1)).squeeze(1)
        source_width = torch.clamp(source_upper - source_lower, min=1e-8)
        rel_pos = (x_vals - source_lower) / source_width

        inverse_perm_spatial = inverse_perm[:, :, None, None, None].expand(-1, -1, d, h, w)
        dest_chunk_idx = torch.gather(inverse_perm_spatial, 1, bin_idx.unsqueeze(1)).squeeze(1)

        dest_lower = torch.gather(dense_edges, 1, dest_chunk_idx.unsqueeze(1)).squeeze(1)
        dest_upper = torch.gather(dense_edges, 1, (dest_chunk_idx + 1).unsqueeze(1)).squeeze(1)
        mapped_scalar = (dest_lower + rel_pos * (dest_upper - dest_lower)).clamp(0.0, 1.0)

        mapped_img = mapped_scalar.unsqueeze(1).expand(-1, c, -1, -1, -1)
        bg_mask = input_images <= dark_threshold
        guidance_map = torch.where(bg_mask, input_images, mapped_img)
        target_hist = hist_module(guidance_map)

        return target_hist, perms_tensor, guidance_map


class V15GridTargetGenerator(BaseTargetGenerator):
    """Non-monotonic local chunk target mapping with strict background masking."""

    def __init__(
        self,
        grid_size: tuple[int, int, int] = (4, 4, 4),
        background_threshold: float = 0.01,
    ):
        super().__init__()
        self.grid_size = tuple(int(v) for v in grid_size)
        self.background_threshold = float(background_threshold)

    def forward(
        self,
        input_images: torch.Tensor,
        num_bins: int,
        num_chunks: int,
        dark_threshold: float,
        hist_module: HistogramModuleLike,
        return_guidance_map: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if input_images.ndim != 5:
            raise ValueError("input_images must be a 5D tensor shaped as (B, C, D, H, W).")

        b, c, d, h, w = input_images.shape
        gz, gy, gx = self.grid_size
        if gz <= 0 or gy <= 0 or gx <= 0:
            raise ValueError("grid_size values must be positive integers.")

        x_scalar = input_images.mean(dim=1, keepdim=True)

        pad_d = (gz - (d % gz)) % gz
        pad_h = (gy - (h % gy)) % gy
        pad_w = (gx - (w % gx)) % gx
        if pad_d or pad_h or pad_w:
            x_scalar_pad = F.pad(x_scalar, (0, pad_w, 0, pad_h, 0, pad_d), mode="replicate")
        else:
            x_scalar_pad = x_scalar

        d_pad, h_pad, w_pad = x_scalar_pad.shape[-3:]
        bd = d_pad // gz
        bh = h_pad // gy
        bw = w_pad // gx

        block_values = (
            x_scalar_pad.view(b, 1, gz, bd, gy, bh, gx, bw)
            .permute(0, 2, 4, 6, 1, 3, 5, 7)
            .reshape(b, gz * gy * gx, -1)
        )

        block_values = block_values.clone()
        block_values[block_values <= dark_threshold] = float("nan")

        q_probs = torch.linspace(0.0, 1.0, num_chunks + 1, device=input_images.device, dtype=torch.float32)
        local_edges = torch.nanquantile(block_values, q_probs, dim=-1).to(input_images.dtype).permute(1, 0, 2)

        local_edges = torch.nan_to_num(local_edges, nan=float(dark_threshold))
        local_edges[:, 0, :] = torch.clamp(local_edges[:, 0, :], max=dark_threshold)
        local_edges[:, -1, :] = torch.clamp(local_edges[:, -1, :], min=1.0)

        local_edges_grid = local_edges.view(b, num_chunks + 1, gz, gy, gx)
        dense_edges = F.interpolate(
            local_edges_grid,
            size=(d, h, w),
            mode="trilinear",
            align_corners=True,
        )
        dense_edges = torch.cummax(dense_edges, dim=1).values

        x_vals = x_scalar.squeeze(1)
        bin_idx = torch.sum(x_vals.unsqueeze(1) > dense_edges, dim=1) - 1
        bin_idx = torch.clamp(bin_idx, 0, num_chunks - 1)

        random_targets = _shared_rand((b, num_chunks), device=input_images.device, dtype=input_images.dtype)
        target_vals = torch.gather(random_targets, 1, bin_idx.reshape(b, -1)).reshape(b, d, h, w)

        mapped_img = target_vals.unsqueeze(1).expand(-1, c, -1, -1, -1)
        guidance_map = torch.where(
            input_images > self.background_threshold,
            mapped_img,
            torch.zeros_like(mapped_img),
        )
        guidance_map = guidance_map.clamp(0.0, 1.0)

        target_hist = hist_module(guidance_map)
        return target_hist, random_targets, guidance_map


class V17MicroAnchorTargetGenerator(BaseTargetGenerator):
    """Micro-anchor guidance generation via 1D histogram peak anchoring and RBF mapping."""

    def __init__(
        self,
        tau: float = 0.05,
        num_peaks: int = 4,
        background_threshold: float = 0.01,
    ):
        super().__init__()
        self.tau = float(tau)
        self.num_peaks = int(num_peaks)
        self.background_threshold = float(background_threshold)

    def forward(
        self,
        input_images: torch.Tensor,
        num_bins: int,
        num_chunks: int,
        dark_threshold: float,
        hist_module: HistogramModuleLike,
        return_guidance_map: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if input_images.ndim != 5:
            raise ValueError("input_images must be a 5D tensor shaped as (B, C, D, H, W).")

        b, c, d, h, w = input_images.shape
        x_scalar = input_images.mean(dim=1, keepdim=True).float()

        hist_1d = hist_module(x_scalar)[:, 0, :]

        sigma = 1.0
        kernel_size = 7
        coords = torch.arange(kernel_size, device=hist_1d.device, dtype=hist_1d.dtype) - (kernel_size - 1) / 2.0
        g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        g = g / g.sum().clamp_min(torch.finfo(g.dtype).eps)
        h_smooth = F.conv1d(
            hist_1d.unsqueeze(1),
            g.view(1, 1, kernel_size),
            padding=kernel_size // 2,
        ).squeeze(1)

        left = F.pad(h_smooth[:, :-1], (1, 0), mode="replicate")
        right = F.pad(h_smooth[:, 1:], (0, 1), mode="replicate")
        is_local_max = (h_smooth >= left) & (h_smooth >= right)
        neg_inf = torch.full_like(h_smooth, torch.finfo(h_smooth.dtype).min)
        peak_scores = torch.where(is_local_max, h_smooth, neg_inf)
        peak_indices = torch.topk(peak_scores, k=self.num_peaks, dim=1).indices

        min_v = float(hist_module.min_value)
        max_v = float(hist_module.max_value)
        bins_minus_one = max(int(hist_module.num_bins) - 1, 1)
        centers = min_v + (peak_indices.to(h_smooth.dtype) / float(bins_minus_one)) * (max_v - min_v)

        x_flat = x_scalar.reshape(b, -1)
        diff = x_flat.unsqueeze(-1) - centers.unsqueeze(1)
        logits = -((diff * diff) / self.tau)
        weights = torch.softmax(logits, dim=-1)

        mu = _shared_rand((b, self.num_peaks), device=input_images.device, dtype=x_flat.dtype)

        synth_flat = torch.sum(weights * mu.unsqueeze(1), dim=-1)
        synth_scalar = synth_flat.reshape(b, 1, d, h, w)
        mapped_img = synth_scalar.expand(-1, c, -1, -1, -1)

        guidance_map = torch.where(
            input_images > self.background_threshold,
            mapped_img,
            torch.zeros_like(mapped_img),
        )
        guidance_map = guidance_map.clamp(0.0, 1.0).to(dtype=input_images.dtype)

        target_hist = hist_module(guidance_map)
        return target_hist, mu.to(dtype=input_images.dtype), guidance_map


class V18SpatialBezierTargetGenerator(BaseTargetGenerator):
    """Spatially varying non-monotonic Bezier intensity field for v18 guidance."""

    def __init__(
        self,
        coarse_grid_size: tuple[int, int, int] = (4, 4, 4),
        background_threshold: float = 0.01,
    ):
        super().__init__()
        self.coarse_grid_size = tuple(int(v) for v in coarse_grid_size)
        self.background_threshold = float(background_threshold)

    def forward(
        self,
        input_images: torch.Tensor,
        num_bins: int,
        num_chunks: int,
        dark_threshold: float,
        hist_module: HistogramModuleLike,
        return_guidance_map: bool = True,
        masks: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del num_bins, num_chunks, dark_threshold, masks

        if "images" in kwargs and kwargs["images"] is not None:
            input_images = kwargs["images"]  # supports __call__(images=..., masks=...)

        if input_images.ndim != 5:
            raise ValueError("input_images must be a 5D tensor shaped as (B, C, D, H, W).")

        b, c, d, h, w = input_images.shape
        gz, gy, gx = self.coarse_grid_size
        if (gz, gy, gx) != (4, 4, 4):
            raise ValueError("V18SpatialBezierTargetGenerator requires coarse_grid_size=(4, 4, 4).")

        coarse_shape = (b, 1, gz, gy, gx)
        p0_coarse = _shared_rand_intensity(coarse_shape, device=input_images.device, dtype=input_images.dtype)
        p1_coarse = _shared_rand_intensity(coarse_shape, device=input_images.device, dtype=input_images.dtype)
        p2_coarse = _shared_rand_intensity(coarse_shape, device=input_images.device, dtype=input_images.dtype)
        p3_coarse = _shared_rand_intensity(coarse_shape, device=input_images.device, dtype=input_images.dtype)

        interp_size = (d, h, w)
        p0 = F.interpolate(p0_coarse, size=interp_size, mode="trilinear", align_corners=True)
        p1 = F.interpolate(p1_coarse, size=interp_size, mode="trilinear", align_corners=True)
        p2 = F.interpolate(p2_coarse, size=interp_size, mode="trilinear", align_corners=True)
        p3 = F.interpolate(p3_coarse, size=interp_size, mode="trilinear", align_corners=True)

        x = input_images.mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        original_dtype = input_images.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            x32 = x.to(torch.float32)
            p032 = p0.to(torch.float32)
            p132 = p1.to(torch.float32)
            p232 = p2.to(torch.float32)
            p332 = p3.to(torch.float32)

            one_minus = 1.0 - x32
            y32 = (
                (one_minus ** 3) * p032
                + 3.0 * (one_minus ** 2) * x32 * p132
                + 3.0 * one_minus * (x32 ** 2) * p232
                + (x32 ** 3) * p332
            )

        y = y32.clamp(0.0, 1.0).to(dtype=original_dtype)
        synthesized_targets = y.expand(-1, c, -1, -1, -1).clone()
        synthesized_targets[input_images < self.background_threshold] = 0.0

        target_hist = hist_module(synthesized_targets)
        control_points = torch.stack([p0_coarse, p1_coarse, p2_coarse, p3_coarse], dim=1)
        return target_hist, control_points, synthesized_targets


class V18_1QuantileAnchoredBezierTargetGenerator(BaseTargetGenerator):
    """Quantile-anchored spatial Bezier mapping over empirical CDF ranks for v18.1."""

    def __init__(
        self,
        coarse_grid_size: tuple[int, int, int] = (4, 4, 4),
        background_threshold: float = 0.01,
        num_quantiles: int = 100,
        max_sample_size: int = 100000,
    ):
        super().__init__()
        self.coarse_grid_size = tuple(int(v) for v in coarse_grid_size)
        self.background_threshold = float(background_threshold)
        self.num_quantiles = int(num_quantiles)
        self.max_sample_size = int(max_sample_size)

    def forward(
        self,
        input_images: torch.Tensor,
        num_bins: int,
        num_chunks: int,
        dark_threshold: float,
        hist_module: HistogramModuleLike,
        return_guidance_map: bool = True,
        masks: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del num_bins, num_chunks, dark_threshold, masks

        if "images" in kwargs and kwargs["images"] is not None:
            input_images = kwargs["images"]  # supports __call__(images=..., masks=...)

        if input_images.ndim != 5:
            raise ValueError("input_images must be a 5D tensor shaped as (B, C, D, H, W).")
        if self.num_quantiles < 2:
            raise ValueError("num_quantiles must be >= 2 for rank mapping.")

        b, c, d, h, w = input_images.shape
        gz, gy, gx = self.coarse_grid_size
        if (gz, gy, gx) != (4, 4, 4):
            raise ValueError("V18_1QuantileAnchoredBezierTargetGenerator requires coarse_grid_size=(4, 4, 4).")

        x = input_images.mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        x_flat = x.view(b, -1)

        total_voxels = x_flat.shape[1]
        sample_stride = max(1, total_voxels // self.max_sample_size) if total_voxels > self.max_sample_size else 1
        sampled = x_flat[:, ::sample_stride].clone()
        sampled[sampled <= self.background_threshold] = float("nan")

        q_probs = torch.linspace(0.0, 1.0, self.num_quantiles, device=input_images.device, dtype=torch.float32)
        quantiles = torch.nanquantile(sampled, q_probs, dim=1).transpose(0, 1)
        quantiles = torch.nan_to_num(quantiles, nan=float(self.background_threshold))
        quantiles[:, 0] = torch.clamp(quantiles[:, 0], max=self.background_threshold)
        quantiles[:, -1] = torch.clamp(quantiles[:, -1], min=1.0)

        rank_idx = torch.searchsorted(quantiles, x_flat, right=True)
        rank_idx = rank_idx.clamp(min=0, max=self.num_quantiles - 1)
        r = rank_idx.to(dtype=x.dtype) / float(self.num_quantiles - 1)
        r = r.view(b, 1, d, h, w)

        invert_mask = (_shared_rand_intensity((b, 1, 1, 1, 1), device=input_images.device, dtype=x.dtype) < 0.5)
        p0 = torch.where(invert_mask, torch.ones_like(r[:, :, :1, :1, :1]), torch.zeros_like(r[:, :, :1, :1, :1]))
        p3 = torch.where(invert_mask, torch.zeros_like(r[:, :, :1, :1, :1]), torch.ones_like(r[:, :, :1, :1, :1]))

        coarse_shape = (b, 1, gz, gy, gx)
        p1_coarse = _shared_rand_intensity(coarse_shape, device=input_images.device, dtype=x.dtype)
        p2_coarse = _shared_rand_intensity(coarse_shape, device=input_images.device, dtype=x.dtype)
        p1 = F.interpolate(p1_coarse, size=(d, h, w), mode="trilinear", align_corners=True)
        p2 = F.interpolate(p2_coarse, size=(d, h, w), mode="trilinear", align_corners=True)

        original_dtype = input_images.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            r32 = r.to(torch.float32)
            p032 = p0.to(torch.float32)
            p132 = p1.to(torch.float32)
            p232 = p2.to(torch.float32)
            p332 = p3.to(torch.float32)

            one_minus = 1.0 - r32
            y32 = (
                (one_minus ** 3) * p032
                + 3.0 * (one_minus ** 2) * r32 * p132
                + 3.0 * one_minus * (r32 ** 2) * p232
                + (r32 ** 3) * p332
            )

        y = y32.clamp(0.0, 1.0).to(dtype=original_dtype)
        y[input_images.mean(dim=1, keepdim=True) < self.background_threshold] = 0.0
        synthesized_targets = y.expand(-1, c, -1, -1, -1).clone()

        target_hist = hist_module(synthesized_targets)
        anchor_state = invert_mask.to(dtype=input_images.dtype).view(b, 1)
        return target_hist, anchor_state, synthesized_targets


class V18_2PiecewiseSplineTargetGenerator(BaseTargetGenerator):
    """Spatially varying piecewise spline mapping over empirical CDF ranks for v18_2."""

    def __init__(
        self,
        coarse_grid_size: tuple[int, int, int] = (8, 8, 8),
        background_threshold: float = 0.01,
        num_quantiles: int = 100,
        max_sample_size: int = 100000,
    ):
        super().__init__()
        self.coarse_grid_size = tuple(int(v) for v in coarse_grid_size)
        self.background_threshold = float(background_threshold)
        self.num_quantiles = int(num_quantiles)
        self.max_sample_size = int(max_sample_size)

    def forward(
        self,
        input_images: torch.Tensor,
        num_bins: int,
        num_chunks: int,
        dark_threshold: float,
        hist_module: HistogramModuleLike,
        return_guidance_map: bool = True,
        masks: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del num_bins, num_chunks, dark_threshold, masks, return_guidance_map

        if "images" in kwargs and kwargs["images"] is not None:
            input_images = kwargs["images"]

        if input_images.ndim != 5:
            raise ValueError("input_images must be a 5D tensor shaped as (B, C, D, H, W).")
        if self.num_quantiles < 2:
            raise ValueError("num_quantiles must be >= 2 for rank mapping.")
        if self.coarse_grid_size != (8, 8, 8):
            raise ValueError("V18_2PiecewiseSplineTargetGenerator requires coarse_grid_size=(8, 8, 8).")

        b, c, d, h, w = input_images.shape
        x = input_images.mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        x_flat = x.view(b, -1)

        total_voxels = x_flat.shape[1]
        sample_stride = max(1, total_voxels // self.max_sample_size) if total_voxels > self.max_sample_size else 1
        sampled = x_flat[:, ::sample_stride].clone()
        sampled[sampled <= self.background_threshold] = float("nan")

        q_probs = torch.linspace(0.0, 1.0, self.num_quantiles, device=input_images.device, dtype=torch.float32)
        quantiles = torch.nanquantile(sampled, q_probs, dim=1).transpose(0, 1)
        quantiles = torch.nan_to_num(quantiles, nan=float(self.background_threshold))
        quantiles[:, 0] = torch.clamp(quantiles[:, 0], max=self.background_threshold)
        quantiles[:, -1] = torch.clamp(quantiles[:, -1], min=1.0)

        rank_idx = torch.searchsorted(quantiles, x_flat, right=True)
        rank_idx = rank_idx.clamp(min=0, max=self.num_quantiles - 1)
        r = (rank_idx.to(dtype=x.dtype) / float(self.num_quantiles - 1)).view(b, 1, d, h, w)

        invert_mask = (_shared_rand_intensity((b, 1, 1, 1, 1), device=input_images.device, dtype=x.dtype) < 0.5)
        y0 = torch.where(invert_mask, torch.ones_like(r[:, :, :1, :1, :1]), torch.zeros_like(r[:, :, :1, :1, :1]))
        y5 = torch.where(invert_mask, torch.zeros_like(r[:, :, :1, :1, :1]), torch.ones_like(r[:, :, :1, :1, :1]))

        coarse_shape = (b, 1, 8, 8, 8)
        interior_coarse = [
            _shared_rand_intensity(coarse_shape, device=input_images.device, dtype=x.dtype)
            for _ in range(4)
        ]
        interior_dense = [
            F.interpolate(grid, size=(d, h, w), mode="trilinear", align_corners=True)
            for grid in interior_coarse
        ]

        y0_dense = y0.expand(-1, -1, d, h, w)
        y5_dense = y5.expand(-1, -1, d, h, w)
        knot_targets = torch.cat([y0_dense, *interior_dense, y5_dense], dim=1)

        knot_positions = torch.linspace(0.0, 1.0, 6, device=input_images.device, dtype=torch.float32)
        orig_dtype = input_images.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            r32 = r.to(torch.float32)
            target32 = knot_targets.to(torch.float32)

            scaled = torch.clamp(r32 * 5.0, 0.0, 5.0 - 1e-6)
            left_idx = torch.floor(scaled).to(torch.long)
            right_idx = torch.clamp(left_idx + 1, max=5)

            q_left = knot_positions[left_idx]
            q_right = knot_positions[right_idx]
            t = (r32 - q_left) / (q_right - q_left)

            y_left = torch.gather(target32, dim=1, index=left_idx)
            y_right = torch.gather(target32, dim=1, index=right_idx)
            y32 = (1.0 - t) * y_left + t * y_right

        y = y32.to(dtype=orig_dtype)
        y[x < self.background_threshold] = 0.0
        synthesized_targets = y.expand(-1, c, -1, -1, -1).clone()
        synthesized_targets = synthesized_targets.clamp(0.0, 1.0)

        target_hist = hist_module(synthesized_targets)
        anchor_state = invert_mask.to(dtype=input_images.dtype).view(b, 1)
        return target_hist, anchor_state, synthesized_targets


class V18_3UnanchoredSplineTargetGenerator(BaseTargetGenerator):
    """Fully unanchored free-knot spatial spline mapping over empirical CDF ranks for v18_3."""

    def __init__(
        self,
        coarse_grid_size: tuple[int, int, int] = (8, 8, 8),
        background_threshold: float = 0.01,
        num_quantiles: int = 100,
        max_sample_size: int = 100000,
    ):
        super().__init__()
        self.coarse_grid_size = tuple(int(v) for v in coarse_grid_size)
        self.background_threshold = float(background_threshold)
        self.num_quantiles = int(num_quantiles)
        self.max_sample_size = int(max_sample_size)

    def forward(
        self,
        input_images: torch.Tensor,
        num_bins: int,
        num_chunks: int,
        dark_threshold: float,
        hist_module: HistogramModuleLike,
        return_guidance_map: bool = True,
        masks: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del num_bins, num_chunks, dark_threshold, masks, return_guidance_map

        if "images" in kwargs and kwargs["images"] is not None:
            input_images = kwargs["images"]

        if input_images.ndim != 5:
            raise ValueError("input_images must be a 5D tensor shaped as (B, C, D, H, W).")
        if self.num_quantiles < 2:
            raise ValueError("num_quantiles must be >= 2 for rank mapping.")
        if self.coarse_grid_size != (8, 8, 8):
            raise ValueError("V18_3UnanchoredSplineTargetGenerator requires coarse_grid_size=(8, 8, 8).")

        b, c, d, h, w = input_images.shape
        x = input_images.mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        x_flat = x.view(b, -1)

        total_voxels = x_flat.shape[1]
        sample_stride = max(1, total_voxels // self.max_sample_size) if total_voxels > self.max_sample_size else 1
        sampled = x_flat[:, ::sample_stride].clone()
        sampled[sampled <= self.background_threshold] = float("nan")

        q_probs = torch.linspace(0.0, 1.0, self.num_quantiles, device=input_images.device, dtype=torch.float32)
        quantiles = torch.nanquantile(sampled, q_probs, dim=1).transpose(0, 1)
        quantiles = torch.nan_to_num(quantiles, nan=float(self.background_threshold))
        quantiles = torch.cummax(quantiles, dim=1).values

        rank_idx = torch.searchsorted(quantiles, x_flat, right=True)
        rank_idx = rank_idx.clamp(min=0, max=self.num_quantiles - 1)
        r = (rank_idx.to(dtype=x.dtype) / float(self.num_quantiles - 1)).view(b, 1, d, h, w)

        coarse_knots = _shared_rand((b, 8, 8, 8, 8), device=input_images.device, dtype=input_images.dtype)
        y_knots = F.interpolate(
            coarse_knots,
            size=(d, h, w),
            mode="trilinear",
            align_corners=True,
        )

        orig_dtype = input_images.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            r32 = r.to(torch.float32)
            y_knots32 = y_knots.to(torch.float32)

            k_idx = torch.clamp((r32 * 7.0).long(), 0, 6)
            t = (r32 * 7.0) - k_idx.to(torch.float32)

            y_lower = torch.gather(y_knots32, 1, k_idx)
            y_upper = torch.gather(y_knots32, 1, k_idx + 1)
            y32 = (1.0 - t) * y_lower + t * y_upper

        y = y32.to(dtype=orig_dtype)
        y[x < self.background_threshold] = 0.0
        synthesized_targets = y.expand(-1, c, -1, -1, -1).clone().clamp(0.0, 1.0)

        target_hist = hist_module(synthesized_targets)
        return target_hist, coarse_knots, synthesized_targets


class V18_4CoalescingSplineTargetGenerator(BaseTargetGenerator):
    """Free-knot spline mapping with vectorized knot coalescence for v18_4."""

    def __init__(
        self,
        coarse_grid_size: tuple[int, int, int] = (8, 8, 8),
        background_threshold: float = 0.01,
        num_quantiles: int = 100,
        max_sample_size: int = 100000,
    ):
        super().__init__()
        self.coarse_grid_size = tuple(int(v) for v in coarse_grid_size)
        self.background_threshold = float(background_threshold)
        self.num_quantiles = int(num_quantiles)
        self.max_sample_size = int(max_sample_size)

    def forward(
        self,
        input_images: torch.Tensor,
        num_bins: int,
        num_chunks: int,
        dark_threshold: float,
        hist_module: HistogramModuleLike,
        return_guidance_map: bool = True,
        masks: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del num_bins, num_chunks, dark_threshold, masks, return_guidance_map

        if "images" in kwargs and kwargs["images"] is not None:
            input_images = kwargs["images"]

        if input_images.ndim != 5:
            raise ValueError("input_images must be a 5D tensor shaped as (B, C, D, H, W).")
        if self.num_quantiles < 2:
            raise ValueError("num_quantiles must be >= 2 for rank mapping.")
        if self.coarse_grid_size != (8, 8, 8):
            raise ValueError("V18_4CoalescingSplineTargetGenerator requires coarse_grid_size=(8, 8, 8).")

        b, c, d, h, w = input_images.shape
        x = input_images.mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        x_flat = x.view(b, -1)

        total_voxels = x_flat.shape[1]
        sample_stride = max(1, total_voxels // self.max_sample_size) if total_voxels > self.max_sample_size else 1
        sampled = x_flat[:, ::sample_stride].clone()
        sampled[sampled <= self.background_threshold] = float("nan")

        q_probs = torch.linspace(0.0, 1.0, self.num_quantiles, device=input_images.device, dtype=torch.float32)
        quantiles = torch.nanquantile(sampled, q_probs, dim=1).transpose(0, 1)
        quantiles = torch.nan_to_num(quantiles, nan=float(self.background_threshold))
        quantiles = torch.cummax(quantiles, dim=1).values

        rank_idx = torch.searchsorted(quantiles, x_flat, right=True)
        rank_idx = rank_idx.clamp(min=0, max=self.num_quantiles - 1)
        r = (rank_idx.to(dtype=x.dtype) / float(self.num_quantiles - 1)).view(b, 1, d, h, w)

        y_base = _shared_rand((b, 8, 8, 8, 8), device=input_images.device, dtype=input_images.dtype)
        idx = torch.arange(8, device=input_images.device).view(1, 8, 1, 1, 1).expand(b, 8, 1, 1, 1)
        keep = (_shared_rand((b, 8, 1, 1, 1), device=input_images.device, dtype=input_images.dtype) > 0.4).long()
        keep[:, 0] = 1
        forward_idx = torch.cummax(idx * keep, dim=1)[0]
        forward_idx = forward_idx.expand(-1, -1, 8, 8, 8)
        y_coalesced = torch.gather(y_base, 1, forward_idx)

        y_coalesced = F.interpolate(
            y_coalesced,
            size=(d, h, w),
            mode="trilinear",
            align_corners=True,
        )

        orig_dtype = input_images.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            r32 = r.to(torch.float32)
            y_coalesced32 = y_coalesced.to(torch.float32)

            k_idx = torch.clamp((r32 * 7.0).long(), 0, 6)
            t = (r32 * 7.0) - k_idx.to(torch.float32)

            y_lower = torch.gather(y_coalesced32, 1, k_idx)
            y_upper = torch.gather(y_coalesced32, 1, k_idx + 1)
            y32 = (1.0 - t) * y_lower + t * y_upper

        y = y32.to(dtype=orig_dtype)
        y[x < self.background_threshold] = 0.0
        synthesized_targets = y.expand(-1, c, -1, -1, -1).clone().clamp(0.0, 1.0)

        target_hist = hist_module(synthesized_targets)
        return target_hist, y_base, synthesized_targets


class V18_6TexturePreservingChunkTargetGenerator(BaseTargetGenerator):
    """Discontinuous raw-intensity chunk mapping with texture-preserving residual scaling for v18_6."""

    def __init__(
        self,
        background_threshold: float = 0.01,
        num_chunks: int = 8,
        max_sample_size: int = 100000,
        alpha_min: float = 0.5,
        alpha_max: float = 2.0,
    ):
        super().__init__()
        self.background_threshold = float(background_threshold)
        self.num_chunks = int(num_chunks)
        self.max_sample_size = int(max_sample_size)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)

    def forward(
        self,
        input_images: torch.Tensor,
        num_bins: int,
        num_chunks: int,
        dark_threshold: float,
        hist_module: HistogramModuleLike,
        return_guidance_map: bool = True,
        masks: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del num_bins, dark_threshold, masks, return_guidance_map

        if "images" in kwargs and kwargs["images"] is not None:
            input_images = kwargs["images"]

        if input_images.ndim != 5:
            raise ValueError("input_images must be a 5D tensor shaped as (B, C, D, H, W).")
        if self.num_chunks != 8:
            raise ValueError("V18_6TexturePreservingChunkTargetGenerator requires num_chunks=8.")

        b, c, d, h, w = input_images.shape
        x = input_images.mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        x_flat = x.view(b, -1)

        total_voxels = x_flat.shape[1]
        sample_stride = max(1, total_voxels // self.max_sample_size) if total_voxels > self.max_sample_size else 1
        sampled = x_flat[:, ::sample_stride].clone()
        sampled[sampled <= self.background_threshold] = float("nan")

        q_probs = torch.linspace(
            0.0,
            1.0,
            self.num_chunks + 1,
            device=input_images.device,
            dtype=torch.float32,
        )
        quantiles = torch.nanquantile(sampled, q_probs, dim=1).transpose(0, 1)
        quantiles = torch.nan_to_num(quantiles, nan=float(self.background_threshold))
        quantiles = torch.cummax(quantiles, dim=1).values

        q = quantiles
        q[:, 0] = 0.0
        q[:, -1] = 1.0

        mu = _shared_rand((b, self.num_chunks), device=input_images.device, dtype=x.dtype)
        alpha_u = _shared_rand((b, self.num_chunks), device=input_images.device, dtype=x.dtype)
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * alpha_u

        chunk_idx_flat = torch.searchsorted(q, x_flat, right=True) - 1
        chunk_idx_flat = chunk_idx_flat.clamp(min=0, max=self.num_chunks - 1)
        chunk_idx = chunk_idx_flat.view(b, 1, d, h, w)

        mu_dense = mu.view(b, self.num_chunks, 1, 1, 1)
        alpha_dense = alpha.view(b, self.num_chunks, 1, 1, 1)
        q_lower_dense = q[:, :-1].view(b, self.num_chunks, 1, 1, 1)

        orig_dtype = input_images.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            x32 = x.to(torch.float32)
            mu32 = mu_dense.to(torch.float32).expand(-1, -1, d, h, w)
            alpha32 = alpha_dense.to(torch.float32).expand(-1, -1, d, h, w)
            q32 = q_lower_dense.to(torch.float32).expand(-1, -1, d, h, w)

            mu_gathered = torch.gather(mu32, dim=1, index=chunk_idx)
            alpha_gathered = torch.gather(alpha32, dim=1, index=chunk_idx)
            q_gathered = torch.gather(q32, dim=1, index=chunk_idx)

            y32 = mu_gathered + alpha_gathered * (x32 - q_gathered)
            y32 = y32.clamp(0.0, 1.0)

        y = y32.to(dtype=orig_dtype)
        y[x < self.background_threshold] = 0.0
        synthesized_targets = y.expand(-1, c, -1, -1, -1).clone()

        target_hist = hist_module(synthesized_targets)

        metadata = torch.cat([mu, alpha], dim=1).to(dtype=orig_dtype)
        return target_hist, metadata, synthesized_targets


class V18_7StochasticTargetGenerator(BaseTargetGenerator):
    """Stochastic rank-space target generator with dynamic quantiles and spatial identity leakage for v18_7."""

    def __init__(
        self,
        background_threshold: float = 0.01,
        max_sample_size: int = 100000,
        alpha_min: float = 0.5,
        alpha_max: float = 2.0,
    ):
        super().__init__()
        self.background_threshold = float(background_threshold)
        self.max_sample_size = int(max_sample_size)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)

    def forward(
        self,
        input_images: torch.Tensor,
        num_bins: int,
        num_chunks: int,
        dark_threshold: float,
        hist_module: HistogramModuleLike,
        return_guidance_map: bool = True,
        masks: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del num_bins, num_chunks, dark_threshold, masks, return_guidance_map

        if "images" in kwargs and kwargs["images"] is not None:
            input_images = kwargs["images"]

        if input_images.ndim != 5:
            raise ValueError("input_images must be a 5D tensor shaped as (B, C, D, H, W).")

        b, c, d, h, w = input_images.shape
        raw = input_images.mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        raw_flat = raw.view(b, -1)

        total_voxels = raw_flat.shape[1]
        sample_stride = max(1, total_voxels // self.max_sample_size) if total_voxels > self.max_sample_size else 1
        sampled = raw_flat[:, ::sample_stride].clone()
        sampled[sampled <= self.background_threshold] = float("nan")

        k = int(_shared_randint_intensity(4, 13, (1,), device=input_images.device).item())
        q_probs = torch.linspace(0.0, 1.0, k + 1, device=input_images.device, dtype=torch.float32)
        quantiles = torch.nanquantile(sampled, q_probs, dim=1).transpose(0, 1)
        quantiles = torch.nan_to_num(quantiles, nan=float(self.background_threshold))
        quantiles = torch.cummax(quantiles, dim=1).values
        quantiles[:, 0] = 0.0
        quantiles[:, -1] = 1.0

        mu = _shared_rand((b, k), device=input_images.device, dtype=torch.float32)
        alpha_u = _shared_rand((b, k), device=input_images.device, dtype=torch.float32)
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * alpha_u

        orig_dtype = input_images.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            raw32 = raw.to(torch.float32)
            raw_flat32 = raw32.view(b, -1)
            quantiles32 = quantiles.to(torch.float32)
            mu32 = mu.to(torch.float32)
            alpha32 = alpha.to(torch.float32)

            chunk_idx = torch.searchsorted(quantiles32[:, 1:-1], raw_flat32, right=True)
            chunk_idx = chunk_idx.clamp(min=0, max=k - 1)

            lower_bounds = quantiles32.gather(1, chunk_idx)
            mu_c = mu32.gather(1, chunk_idx)
            alpha_c = alpha32.gather(1, chunk_idx)

            y_synth_flat = mu_c + alpha_c * (raw_flat32 - lower_bounds)
            y_synth_flat = y_synth_flat.clamp(0.0, 1.0)
            y_synth = y_synth_flat.view(b, 1, d, h, w)

            s = int(_shared_randint_intensity(1, 9, (1,), device=input_images.device).item())
            alpha_coarse = _shared_rand((b, 1, s, s, s), device=input_images.device, dtype=torch.float32)
            alpha_dense = F.interpolate(
                alpha_coarse,
                size=(d, h, w),
                mode="trilinear",
                align_corners=s > 1,
            )

            y_final = alpha_dense * raw32 + (1.0 - alpha_dense) * y_synth
            y_final = y_final.clamp(0.0, 1.0)
            y_final = torch.where(raw32 < self.background_threshold, torch.zeros_like(y_final), y_final)

        synthesized_targets = y_final.to(dtype=orig_dtype).expand(-1, c, -1, -1, -1).clone()
        target_hist = hist_module(synthesized_targets)
        metadata = torch.cat([mu, alpha], dim=1).to(dtype=orig_dtype)
        return target_hist, metadata, synthesized_targets


def _create_range_translation_guidance_map(
    input_image: torch.Tensor,
    perms: torch.Tensor,
    num_chunks: int,
    dark_threshold: float,
) -> torch.Tensor:
    b = input_image.shape[0]

    flat_img = input_image.view(b, -1)

    max_sample_size = 100000
    total_voxels = flat_img.shape[1]
    sample_stride = max(1, total_voxels // max_sample_size) if total_voxels > max_sample_size else 1

    flat_sample = flat_img[:, ::sample_stride].clone()
    bg_mask_sample = flat_sample <= dark_threshold
    flat_sample[bg_mask_sample] = float("nan")

    q_probs = torch.linspace(0.0, 1.0, num_chunks + 1, device=input_image.device, dtype=torch.float32)
    edges = torch.nanquantile(flat_sample, q_probs, dim=1).to(input_image.dtype).transpose(0, 1)

    edges[:, -1] = torch.clamp(edges[:, -1], min=1.0)
    edges[:, 0] = torch.clamp(edges[:, 0], max=dark_threshold)

    b_idx = torch.arange(b, device=perms.device).unsqueeze(1)
    chunk_idx = torch.arange(num_chunks, device=perms.device).expand(b, num_chunks)
    inverse_perm = torch.empty_like(perms)
    inverse_perm[b_idx, perms] = chunk_idx

    bin_idx = torch.sum(flat_img.unsqueeze(-1) > edges.unsqueeze(1), dim=-1) - 1
    bin_idx = torch.clamp(bin_idx, 0, num_chunks - 1)

    source_lower = edges[b_idx, bin_idx]
    source_upper = edges[b_idx, bin_idx + 1]
    width = torch.clamp(source_upper - source_lower, min=1e-8)

    rel_pos = (flat_img - source_lower) / width

    dest_chunk_idx = inverse_perm[b_idx, bin_idx]
    dest_lower = edges[b_idx, dest_chunk_idx]
    dest_upper = edges[b_idx, dest_chunk_idx + 1]
    dest_width = dest_upper - dest_lower

    mapped_flat = dest_lower + rel_pos * dest_width
    mapped_flat = mapped_flat.clamp(0.0, 1.0)

    mapped_img = mapped_flat.view_as(input_image)

    bg_mask = input_image <= dark_threshold
    mapped_img = torch.where(bg_mask, input_image, mapped_img)

    return mapped_img
