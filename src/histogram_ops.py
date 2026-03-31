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
    if input_images.ndim != 5:
        raise ValueError("input_images must be a 5D tensor shaped as (B, C, D, H, W).")

    b, c, d, h, w = input_images.shape
    gz, gy, gx = (int(grid_size[0]), int(grid_size[1]), int(grid_size[2]))
    if gz <= 0 or gy <= 0 or gx <= 0:
        raise ValueError("grid_size values must be positive integers.")

    # Per-sample permutation remains global to preserve chunk-level remapping semantics.
    perms = [torch.randperm(num_chunks, device=input_images.device) for _ in range(b)]
    perms_tensor = torch.stack(perms, dim=0)

    # Build local quantiles on a scalar field then broadcast mapping across channels.
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

    # (B, 1, D, H, W) -> (B, G, voxels_per_block)
    block_values = (
        x_scalar_pad
        .view(b, 1, gz, bd, gy, bh, gx, bw)
        .permute(0, 2, 4, 6, 1, 3, 5, 7)
        .reshape(b, gz * gy * gx, -1)
    )

    block_values = block_values.clone()
    block_values[block_values <= dark_threshold] = float("nan")

    q_probs = torch.linspace(0.0, 1.0, num_chunks + 1, device=input_images.device, dtype=torch.float32)
    local_edges = torch.nanquantile(block_values, q_probs, dim=-1).to(input_images.dtype).permute(1, 0, 2)

    # Guard all-background blocks and preserve boundary constraints.
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
    # Keep chunk edges monotonic after interpolation.
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


def generate_non_monotonic_grid_targets(
    input_images: torch.Tensor,
    num_chunks: int,
    dark_threshold: float,
    hist_module: DifferentiableHistogram3D,
    grid_size: tuple[int, int, int] = (4, 4, 4),
    background_threshold: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create v15 non-monotonic spatial chunk targets with strict background masking."""
    if input_images.ndim != 5:
        raise ValueError("input_images must be a 5D tensor shaped as (B, C, D, H, W).")

    b, c, d, h, w = input_images.shape
    gz, gy, gx = (int(grid_size[0]), int(grid_size[1]), int(grid_size[2]))
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
        x_scalar_pad
        .view(b, 1, gz, bd, gy, bh, gx, bw)
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

    # Independent random chunk targets (non-monotonic by design).
    random_targets = torch.rand((b, num_chunks), device=input_images.device, dtype=input_images.dtype)
    target_vals = torch.gather(random_targets, 1, bin_idx.reshape(b, -1)).reshape(b, d, h, w)

    mapped_img = target_vals.unsqueeze(1).expand(-1, c, -1, -1, -1)
    guidance_map = torch.where(input_images > float(background_threshold), mapped_img, torch.zeros_like(mapped_img))
    guidance_map = guidance_map.clamp(0.0, 1.0)

    target_hist = hist_module(guidance_map)
    return target_hist, random_targets, guidance_map


def generate_micro_anchored_targets(
    input_images: torch.Tensor,
    hist_module: DifferentiableHistogram3D,
    tau: float = 0.05,
    num_peaks: int = 4,
    background_threshold: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create v17 micro-anchor guidance via 1D tissue peak clustering and RBF mapping."""
    if input_images.ndim != 5:
        raise ValueError("input_images must be a 5D tensor shaped as (B, C, D, H, W).")

    b, c, d, h, w = input_images.shape
    x_scalar = input_images.mean(dim=1, keepdim=True).float()

    # 1) Differentiable 1D soft histogram over 128 bins using existing optimized path.
    hist_1d = hist_module(x_scalar)[:, 0, :]

    # Smooth histogram with 1D Gaussian kernel (vectorized over batch).
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

    # 2) Local maxima extraction + top-K peaks per sample.
    left = F.pad(h_smooth[:, :-1], (1, 0), mode="replicate")
    right = F.pad(h_smooth[:, 1:], (0, 1), mode="replicate")
    is_local_max = (h_smooth >= left) & (h_smooth >= right)
    neg_inf = torch.full_like(h_smooth, torch.finfo(h_smooth.dtype).min)
    peak_scores = torch.where(is_local_max, h_smooth, neg_inf)
    peak_indices = torch.topk(peak_scores, k=int(num_peaks), dim=1).indices

    # Convert peak indices to intensity anchors in [0, 1] using histogram range.
    min_v = float(hist_module.min_value)
    max_v = float(hist_module.max_value)
    bins_minus_one = max(int(hist_module.num_bins) - 1, 1)
    centers = min_v + (peak_indices.to(h_smooth.dtype) / float(bins_minus_one)) * (max_v - min_v)

    # 3) RBF soft assignment for every voxel to K anchors.
    x_flat = x_scalar.reshape(b, -1)
    diff = x_flat.unsqueeze(-1) - centers.unsqueeze(1)
    logits = -((diff * diff) / float(tau))
    weights = torch.softmax(logits, dim=-1)

    # 4) Independent, unsorted random targets.
    mu = torch.rand((b, int(num_peaks)), device=input_images.device, dtype=x_flat.dtype)

    # 5) Synthesis.
    synth_flat = torch.sum(weights * mu.unsqueeze(1), dim=-1)
    synth_scalar = synth_flat.reshape(b, 1, d, h, w)
    mapped_img = synth_scalar.expand(-1, c, -1, -1, -1)

    # 6) Strict background masking (v15 inheritance).
    guidance_map = torch.where(input_images > float(background_threshold), mapped_img, torch.zeros_like(mapped_img))
    guidance_map = guidance_map.clamp(0.0, 1.0).to(dtype=input_images.dtype)

    target_hist = hist_module(guidance_map)
    return target_hist, mu.to(dtype=input_images.dtype), guidance_map



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

    if str(gen_version) in ("v8", "v9", "v10", "v11"):
        return generate_grid_unified_targets(
            input_images=input_images,
            num_chunks=num_chunks,
            dark_threshold=dark_threshold,
            hist_module=hist_module,
            grid_size=grid_size,
        )

    if str(gen_version) == "v15":
        return generate_non_monotonic_grid_targets(
            input_images=input_images,
            num_chunks=num_chunks,
            dark_threshold=dark_threshold,
            hist_module=hist_module,
            grid_size=grid_size,
            background_threshold=0.01,
        )

    if str(gen_version) == "v17_micro_anchor":
        return generate_micro_anchored_targets(
            input_images=input_images,
            hist_module=hist_module,
            tau=0.05,
            num_peaks=4,
            background_threshold=0.01,
        )

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

    return target_hist, perms_tensor, guidance_map