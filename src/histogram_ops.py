from __future__ import annotations

import math
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
        flat_x = x.reshape(b, c, -1).unsqueeze(2)
        distances = torch.abs(flat_x - self.bin_centers) / (self.bin_width + self.eps)
        weights = torch.clamp(1.0 - distances, min=0.0)

        if mask is not None:
            if mask.shape != x.shape:
                raise ValueError("Mask shape must match the input tensor shape.")
            flat_mask = mask.reshape(b, c, 1, -1).to(dtype=weights.dtype)
            weights = weights * flat_mask

        # Return RAW counts. Do not normalize to PDF here.
        hist = weights.sum(dim=-1) 
        return hist


def create_range_translation_guidance_map(
    input_image: torch.Tensor,
    perms: torch.Tensor,
    num_chunks: int,
    dark_threshold: float,
) -> torch.Tensor:
    b = input_image.shape[0]
    
    # Flatten spatial dims to compute quantiles
    flat_img = input_image.view(b, -1)
    
    # Mask out background by setting to NaN so nanquantile strictly looks at foreground
    flat_fg = flat_img.clone()
    bg_mask_flat = flat_fg <= dark_threshold
    flat_fg[bg_mask_flat] = float('nan')

    q_probs = torch.linspace(0.0, 1.0, num_chunks + 1, device=input_image.device, dtype=torch.float32)
    
    # (num_chunks+1, b) -> (b, num_chunks+1)
    edges = torch.nanquantile(flat_fg, q_probs, dim=1).to(input_image.dtype).transpose(0, 1)

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
    bin_idx = torch.searchsorted(edges, flat_img, right=False) - 1
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
    
    mapped_img = mapped_flat.view_as(input_image)
    
    # Keep background unchanged
    bg_mask = input_image <= dark_threshold
    mapped_img = torch.where(bg_mask, input_image, mapped_img)

    return mapped_img



def generate_unified_targets(
    input_images: torch.Tensor,
    num_bins: int,
    num_chunks: int,
    dark_threshold: float,
    hist_module: DifferentiableHistogram3D,
    return_guidance_map: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create chunk-permuted target histograms.
    NEW: Generates the guidance map internally and derives the histogram directly from it, 
    ensuring 100% synchronization between spatial targets and distribution targets.
    """
    if input_images.ndim != 5:
        raise ValueError("input_images must be a 5D tensor shaped as (B, C, D, H, W).")

    b = input_images.shape[0]
    
    # 1. Generate random permutations
    perms = []
    for _ in range(b):
        perms.append(torch.randperm(num_chunks, device=input_images.device))
    perms_tensor = torch.stack(perms, dim=0)

    # 2. Create the perfect percentile-based guidance map
    with torch.no_grad():
        guidance_map = create_range_translation_guidance_map(
            input_images, 
            perms_tensor, 
            num_chunks, 
            dark_threshold
        )

    # 3. Target histogram is natively just the distribution of the generated guidance map
    target_hist = hist_module(guidance_map)

    if return_guidance_map:
        return target_hist, perms_tensor, guidance_map

    return target_hist, perms_tensor