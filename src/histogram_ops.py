from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def apply_gaussian_blur_3d(
    tensor: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 2.0,
) -> torch.Tensor:
    """Apply depthwise 3D Gaussian blur to a tensor shaped as (B, C, D, H, W)."""
    if tensor.ndim != 5:
        raise ValueError(f"Expected a 5D tensor (B, C, D, H, W), got shape {tuple(tensor.shape)}")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    coords = torch.arange(kernel_size, dtype=tensor.dtype, device=tensor.device) - (kernel_size - 1) / 2.0
    g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1d = g1d / g1d.sum().clamp_min(torch.finfo(g1d.dtype).eps)

    channels = tensor.shape[1]
    padding = kernel_size // 2
    
    # 3D Gaussian Blur is linearly separable into 3 1D convolutions.
    k_d = g1d.view(1, 1, kernel_size, 1, 1).expand(channels, 1, kernel_size, 1, 1).contiguous()
    k_h = g1d.view(1, 1, 1, kernel_size, 1).expand(channels, 1, 1, kernel_size, 1).contiguous()
    k_w = g1d.view(1, 1, 1, 1, kernel_size).expand(channels, 1, 1, 1, kernel_size).contiguous()

    smoothed = F.conv3d(tensor, k_d, padding=(padding, 0, 0), groups=channels)
    smoothed = F.conv3d(smoothed, k_h, padding=(0, padding, 0), groups=channels)
    smoothed = F.conv3d(smoothed, k_w, padding=(0, 0, padding), groups=channels)
    
    return smoothed


class DifferentiableHistogram3D(nn.Module):
    """Differentiable soft histogram for 3D volumes returning RAW VOXEL COUNTS."""

    def __init__(self, num_bins: int = 64, value_range: tuple[float, float] = (0.0, 1.0), eps: float = 1e-8):
        super().__init__()
        self.num_bins = num_bins
        self.min_value = float(value_range[0])
        self.max_value = float(value_range[1])
        self.eps = eps

        bin_centers = torch.linspace(self.min_value, self.max_value, num_bins)
        self.register_buffer("bin_centers", bin_centers.view(1, 1, num_bins, 1), persistent=False)
        self.bin_width = (self.max_value - self.min_value) / max(num_bins - 1, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected a 5D tensor (B, C, D, H, W), got shape {tuple(x.shape)}")

        b, c, *_ = x.shape
        flat_x = x.reshape(b, c, -1)

        # For evenly spaced bins, the triangular kernel is non-zero only for two neighboring bins.
        scaled = (flat_x - self.min_value) / (self.bin_width + self.eps)
        left_idx = torch.floor(scaled).to(torch.long)
        right_idx = left_idx + 1

        wl = (right_idx.to(flat_x.dtype) - scaled).clamp(0.0, 1.0)
        wr = (scaled - left_idx.to(flat_x.dtype)).clamp(0.0, 1.0)

        left_idx = left_idx.clamp(0, self.num_bins - 1)
        right_idx = right_idx.clamp(0, self.num_bins - 1)

        if mask is not None:
            if mask.shape != x.shape:
                raise ValueError("Mask shape must match the input tensor shape.")
            flat_mask = mask.reshape(b, c, -1).to(dtype=flat_x.dtype)
            wl = wl * flat_mask
            wr = wr * flat_mask

        hist = torch.zeros((b, c, self.num_bins), device=x.device, dtype=x.dtype)
        hist.scatter_add_(2, left_idx, wl)
        hist.scatter_add_(2, right_idx, wr)

        # Return RAW counts. Do not normalize to PDF here.
        return hist


def create_range_translation_guidance_map(
    input_image: torch.Tensor,
    perms: torch.Tensor,
    num_chunks: int,
    dark_threshold: float,
) -> torch.Tensor:
    b = input_image.shape[0]
    
    # Reshape to (B, C, -1) for vectorized operations - avoids forced contiguous() allocation
    # Keep memory format natural to avoid expensive aten::copy_ from channels_last<->contiguous conversions
    flat_img = input_image.view(b, -1)
    
    # 1. Algorithmic Complexity: Strided Spatial Subsampling to avoid O(N log N) on full volume
    # Limit quantile computation to max 100k voxels for speed
    max_sample_size = 100000
    total_voxels = flat_img.shape[1]
    sample_stride = max(1, total_voxels // max_sample_size) if total_voxels > max_sample_size else 1
    
    # Clone ONLY the sample subset since we'll modify it with NaN
    flat_sample = flat_img[:, ::sample_stride].clone()
    
    # Mask out background by setting to NaN so nanquantile strictly looks at foreground
    bg_mask_sample = flat_sample <= dark_threshold
    flat_sample[bg_mask_sample] = float('nan')

    q_probs = torch.linspace(0.0, 1.0, num_chunks + 1, device=input_image.device, dtype=torch.float32)

    # (num_chunks+1, b) -> (b, num_chunks+1)
    # nanquantile creates new tensor, transpose creates view - no clone needed
    edges = torch.nanquantile(flat_sample, q_probs, dim=1).to(input_image.dtype).transpose(0, 1)

    edges[:, -1] = torch.clamp(edges[:, -1], min=1.0)
    # Ensure background lower bound is at least dark_threshold (or min fg)
    edges[:, 0] = torch.clamp(edges[:, 0], max=dark_threshold)

    # Inverse permutation vectorized
    b_idx = torch.arange(b, device=perms.device).unsqueeze(1)
    chunk_idx = torch.arange(num_chunks, device=perms.device).expand(b, num_chunks)
    inverse_perm = torch.empty_like(perms)
    inverse_perm[b_idx, perms] = chunk_idx

    # Vectorized bin assignment
    # Use right=False so that values equal to an edge go to the right bin, except for the max.
    # Bypassing Inductor's torch.searchsorted PermuteView crash on 3D data formats via broadcasting equality logic.
    bin_idx = torch.sum(flat_img.unsqueeze(-1) > edges.unsqueeze(1), dim=-1) - 1
    # clamp to [0, num_chunks - 1] covers edge cases (e.g. max val)
    bin_idx = torch.clamp(bin_idx, 0, num_chunks - 1)
    
    # Source boundaries
    source_lower = edges[b_idx, bin_idx]
    source_upper = edges[b_idx, bin_idx + 1]
    width = torch.clamp(source_upper - source_lower, min=1e-8)
    
    # Relative pos
    rel_pos = (flat_img - source_lower) / width
    
    # Destination bins
    dest_chunk_idx = inverse_perm[b_idx, bin_idx]
    dest_lower = edges[b_idx, dest_chunk_idx]
    dest_upper = edges[b_idx, dest_chunk_idx + 1]
    dest_width = dest_upper - dest_lower
    
    mapped_flat = dest_lower + rel_pos * dest_width
    mapped_flat = mapped_flat.clamp(0.0, 1.0)
    
    # Reshape back to original spatial dimensions without forcing memory format
    mapped_img = mapped_flat.view_as(input_image)
    
    # Keep background unchanged
    bg_mask = input_image <= dark_threshold
    mapped_img = torch.where(bg_mask, input_image, mapped_img)

    return mapped_img


def generate_grid_unified_targets(
    input_images: torch.Tensor,
    num_chunks: int,
    dark_threshold: float,
    hist_module: DifferentiableHistogram3D,
    grid_size: tuple[int, int, int] = (4, 4, 4),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create v8 spatially-varying chunk targets with trilinear-interpolated local quantiles."""
    from src.target_generators import V8GridTargetGenerator

    generator = V8GridTargetGenerator(grid_size=grid_size)
    return generator(
        input_images=input_images,
        num_bins=int(hist_module.num_bins),
        num_chunks=num_chunks,
        dark_threshold=dark_threshold,
        hist_module=hist_module,
    )


def generate_non_monotonic_grid_targets(
    input_images: torch.Tensor,
    num_chunks: int,
    dark_threshold: float,
    hist_module: DifferentiableHistogram3D,
    grid_size: tuple[int, int, int] = (4, 4, 4),
    background_threshold: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create v15 non-monotonic spatial chunk targets with strict background masking."""
    from src.target_generators import V15GridTargetGenerator

    generator = V15GridTargetGenerator(
        grid_size=grid_size,
        background_threshold=background_threshold,
    )
    return generator(
        input_images=input_images,
        num_bins=int(hist_module.num_bins),
        num_chunks=num_chunks,
        dark_threshold=dark_threshold,
        hist_module=hist_module,
    )


def generate_micro_anchored_targets(
    input_images: torch.Tensor,
    hist_module: DifferentiableHistogram3D,
    tau: float = 0.05,
    num_peaks: int = 4,
    background_threshold: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create v17 micro-anchor guidance via 1D tissue peak clustering and RBF mapping."""
    from src.target_generators import V17MicroAnchorTargetGenerator

    generator = V17MicroAnchorTargetGenerator(
        tau=tau,
        num_peaks=num_peaks,
        background_threshold=background_threshold,
    )
    return generator(
        input_images=input_images,
        num_bins=int(hist_module.num_bins),
        num_chunks=num_peaks,
        dark_threshold=dark_threshold,
        hist_module=hist_module,
    )



def generate_unified_targets(
    input_images: torch.Tensor,
    num_bins: int,
    num_chunks: int,
    dark_threshold: float,
    hist_module: DifferentiableHistogram3D,
    return_guidance_map: bool = True,
    gen_version: str | None = None,
    grid_size: tuple[int, int, int] = (4, 4, 4),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create chunk-permuted target histograms.
    ALWAYS returns: (target_hist, perms, guidance_map)
    Ensures 100% synchronization between spatial targets and distribution targets.
    """
    if input_images.ndim != 5:
        raise ValueError("input_images must be a 5D tensor shaped as (B, C, D, H, W).")

    from src.target_generators import (
        LegacyChunkTargetGenerator,
        V8GridTargetGenerator,
        V15GridTargetGenerator,
        V17MicroAnchorTargetGenerator,
    )

    if str(gen_version) in ("v8", "v9", "v10", "v11"):
        generator = V8GridTargetGenerator(grid_size=grid_size)
    elif str(gen_version) == "v15":
        generator = V15GridTargetGenerator(grid_size=grid_size, background_threshold=0.01)
    elif str(gen_version) == "v17_micro_anchor":
        generator = V17MicroAnchorTargetGenerator(tau=0.05, num_peaks=4, background_threshold=0.01)
    else:
        generator = LegacyChunkTargetGenerator()

    return generator(
        input_images=input_images,
        num_bins=num_bins,
        num_chunks=num_chunks,
        dark_threshold=dark_threshold,
        hist_module=hist_module,
        return_guidance_map=return_guidance_map,
    )