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
    g3d = torch.einsum("i,j,k->ijk", g1d, g1d, g1d)

    channels = tensor.shape[1]
    kernel = g3d.view(1, 1, kernel_size, kernel_size, kernel_size).repeat(channels, 1, 1, 1, 1)
    padding = kernel_size // 2
    return F.conv3d(tensor, kernel, padding=padding, groups=channels)


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
    """
    Project the chunk permutation back into voxel space to obtain a guidance map.
    NEW: Uses Percentile/Quantile chunking to group anatomical tissues together
    based on equal voxel mass rather than uniform intensity widths.
    """
    if input_image.ndim != 5:
        raise ValueError("input_image must be a 5D tensor shaped as (B, C, D, H, W).")

    guidance_map = input_image.clone()
    b = input_image.shape[0]

    for i in range(b):
        img = input_image[i]
        perm = perms[i]
        
        fg_mask = img > dark_threshold
        fg_vals = img[fg_mask]

        if fg_vals.numel() < num_chunks:
            continue

        # ---> FIX: SUB-SAMPLE TO SPEED UP GPU SORTING <---
        max_samples = 100_000
        if fg_vals.numel() > max_samples:
            # Grab a random subset of 100k voxels
            indices = torch.randperm(fg_vals.numel(), device=img.device)[:max_samples]
            sample_vals = fg_vals[indices]
        else:
            sample_vals = fg_vals

        # 1. Calculate quantile boundaries on the tiny sample
        q_probs = torch.linspace(0.0, 1.0, num_chunks + 1, device=input_image.device, dtype=torch.float32)
        edges = torch.quantile(sample_vals.float(), q_probs).to(img.dtype)
        
        # Cap limits to prevent out of bounds
        edges[-1] = max(edges[-1], 1.0)
        edges[0] = min(edges[0], dark_threshold)

        inverse_perm = torch.empty_like(perm)
        inverse_perm[perm] = torch.arange(num_chunks, device=input_image.device)

        mapped_img = img.clone()
        
        # 2. Map values chunk by chunk
        for chunk_idx in range(num_chunks):
            lower = edges[chunk_idx]
            upper = edges[chunk_idx + 1]
            
            # Create mask for voxels falling into this percentile bucket
            if chunk_idx == num_chunks - 1:
                c_mask = fg_mask & (img >= lower) & (img <= upper)
            else:
                c_mask = fg_mask & (img >= lower) & (img < upper)
            
            c_vals = img[c_mask]
            if c_vals.numel() == 0:
                continue
                
            # Relative position within the source chunk [0.0 to 1.0]
            width = upper - lower
            if width <= 0:
                width = 1e-8 # Prevent division by zero if many voxels share exact same intensity
            rel_pos = (c_vals - lower) / width
            
            # Find destination chunk bounds
            dest_idx = inverse_perm[chunk_idx]
            dest_lower = edges[dest_idx]
            dest_upper = edges[dest_idx + 1]
            dest_width = dest_upper - dest_lower
            
            # Apply relative position to destination chunk
            mapped_vals = dest_lower + rel_pos * dest_width
            mapped_img[c_mask] = mapped_vals.clamp(0.0, 1.0)
            
        guidance_map[i] = mapped_img

    return guidance_map


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