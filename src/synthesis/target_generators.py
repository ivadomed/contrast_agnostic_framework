from __future__ import annotations

from typing import List, Optional, Protocol

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from src.synthesis.intensity_ops import _shared_rand as _shared_rand_intensity
from src.synthesis.intensity_ops import _shared_randint as _shared_randint_intensity
from src.synthesis.noise_ops import generate_fractal_noise_3d


_SHARED_RNG_COUNTER = 0


def _next_shared_seed() -> int:
    global _SHARED_RNG_COUNTER
    _SHARED_RNG_COUNTER += 1
    seed = (int(torch.initial_seed()) + _SHARED_RNG_COUNTER) % (2**63 - 1)
    if dist.is_available() and dist.is_initialized():
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        seed_tensor = torch.tensor([seed], dtype=torch.long, device=device)
        dist.broadcast(seed_tensor, src=0)
        seed = int(seed_tensor.item())
    return seed


def _normalize_guidance(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sample min-max normalization within brain mask.
    Preserves all relative intensities from affine remapping (no hard clamp saturation).
    To revert to hard clamp: replace calls with `torch.clamp(y, 0.0, 1.0)`.
    """
    y_min = (y * mask).amin(dim=(2, 3, 4), keepdim=True)
    y_max = (y * mask).amax(dim=(2, 3, 4), keepdim=True)
    y = (y - y_min) / (y_max - y_min + 1e-6)
    return y


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
        labels: torch.Tensor | None = None,
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
        labels: torch.Tensor | None = None,
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
        labels: torch.Tensor | None = None,
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
        labels: torch.Tensor | None = None,
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
        labels: torch.Tensor | None = None,
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


class V19LabelConditionedTextureGenerator(BaseTargetGenerator):
    """
    V19 Stochastic Semantic Decoupling: Merges geometric label-priors with
    texture-preserving latent space.
    """

    def __init__(self, label_classes: Optional[List[int]] = None):
        super().__init__()
        self.label_classes: List[int] = label_classes if label_classes is not None else [1, 2, 3]

    def __call__(
        self,
        input_images: torch.Tensor,
        hist_module: nn.Module,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = input_images
        B, C, D, H, W = images.shape
        device = images.device
        dtype = images.dtype

        # Step A: Base v18_6 Background Synthesis
        mask = images > 0.01
        
        y = images.clone()
        
        # Subsample to compute K=8 quantile edges
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=False):
            images_f = images.float()
            
            # Generate random base targets (or use pre-supplied quasi-random values)
            _mu_lhc = kwargs.get("mu_lhc", None)
            _alpha_lhc_raw = kwargs.get("alpha_lhc_raw", None)
            if _mu_lhc is not None:
                mu_base = torch.as_tensor(_mu_lhc, device=device, dtype=torch.float32).view(B, 8)
            else:
                mu_base = _shared_rand((B, 8), device=device, dtype=torch.float32)
            if _alpha_lhc_raw is not None:
                alpha_base = torch.as_tensor(_alpha_lhc_raw, device=device, dtype=torch.float32).view(B, 8) * 1.5 + 0.5
            else:
                alpha_base = _shared_rand((B, 8), device=device, dtype=torch.float32) * 1.5 + 0.5
            
            q_edges = torch.linspace(0, 1, 9, device=device)
            
            # Bucketize
            c_i = torch.bucketize(images_f, q_edges) - 1
            c_i = torch.clamp(c_i, 0, 7)
            
            mu_c = mu_base.view(B, 8, 1, 1, 1).expand(B, 8, D, H, W).gather(1, c_i)
            alpha_c = alpha_base.view(B, 8, 1, 1, 1).expand(B, 8, D, H, W).gather(1, c_i)
            q_c_lower = q_edges[:-1].view(1, 8, 1, 1, 1).expand(B, 8, D, H, W).gather(1, c_i)
            q_c_upper = q_edges[1:].view(1, 8, 1, 1, 1).expand(B, 8, D, H, W).gather(1, c_i)
            q_c_center = (q_c_lower + q_c_upper) * 0.5

            y_base = mu_c + alpha_c * (images_f - q_c_center)
            y = torch.where(mask, y_base.to(dtype), y)

        # Step B: Stochastic Semantic Decoupling
        if labels is not None:
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=False):
                if labels.dim() == 4:
                    labels = labels.unsqueeze(1)
                if any(dim <= 0 for dim in labels.shape[2:]):
                    labels = None
                else:
                    if labels.shape[2:] != images.shape[2:]:
                        labels = F.interpolate(labels.float(), size=images.shape[2:], mode="nearest")
                    labels = labels.to(device=device)
                    y_f = y.float()
                    images_f = images.float()
                    
                    for c in self.label_classes:
                        # DDP-safe boolean decoupling mask per batch item
                        decouple = _shared_rand((B, 1, 1, 1, 1), device=device, dtype=torch.float32) > 0.5
                        
                        mu_path = _shared_rand((B, 1, 1, 1, 1), device=device, dtype=torch.float32)
                        alpha_path = _shared_rand((B, 1, 1, 1, 1), device=device, dtype=torch.float32) * 1.5 + 0.5
                        
                        class_mask = (labels == c)
                        
                        # Calculate mean intensity of class voxels per batch item
                        # Use safe division
                        class_sum = (images_f * class_mask).sum(dim=(1, 2, 3, 4), keepdim=True)
                        class_count = class_mask.sum(dim=(1, 2, 3, 4), keepdim=True)
                        class_count_safe = torch.clamp(class_count, min=1.0)
                        mean_c = class_sum / class_count_safe
                        
                        y_override = mu_path + alpha_path * (images_f - mean_c)
                        
                        # Apply override conditionally
                        valid_override = class_mask & decouple & (class_count > 0)
                        y_f = torch.where(valid_override, y_override, y_f)
                        
                    y = y_f.to(dtype)
                
        # Step C: Normalize & Mask (v26_4+: min-max within mask; revert: torch.clamp(y,0,1))
        y = _normalize_guidance(y.float(), mask.float()).to(dtype)
        y = torch.where(mask, y, torch.zeros_like(y))

        target_hist = hist_module(y)
        return target_hist, y, y

class V23RandomChunkTargetGenerator(BaseTargetGenerator):
    """
    V23: V19 chunk remap with K drawn randomly each forward pass from a
    log-spaced discrete set {2, 3, 4, 6, 8, 12, 16}.

    Rationale: with fixed K=8 the probability of drawing mu values that
    produce T2w-like contrast (all WM chunks low AND all GM chunks high)
    is ~(0.4)^3 × (0.5)^3 ≈ 0.7%.  With K=2 a single draw controls the
    entire WM bin, raising P(T2w-like) to ~16%.  Randomising K across the
    set gives ~2–3× more T2w-favorable guidance maps in expectation, and up
    to ~16% when K=2 is drawn — enough for the network to learn the mapping.

    LHC inference: mu_lhc and alpha_lhc_raw are accepted as 8-dim vectors
    (Sobol d=16 unchanged).  For K≤8 the first K values are used; for K>8
    the 8-dim vector is tiled to fill K positions.  K is drawn fresh each
    forward pass (categorical, DDP-safe).
    """

    K_CHOICES: list[int] = [2, 3, 4, 6, 8, 12, 16]

    def __call__(
        self,
        input_images: torch.Tensor,
        hist_module: nn.Module,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = input_images
        B, C, D, H, W = images.shape
        device = images.device
        dtype = images.dtype

        mask = images > 0.01
        y = images.clone()

        # Draw K (DDP-safe: use _shared_rand so all ranks pick the same K)
        k_idx = int(_shared_rand((1,), device=device, dtype=torch.float32).item()
                    * len(self.K_CHOICES))
        K = self.K_CHOICES[min(k_idx, len(self.K_CHOICES) - 1)]

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                            enabled=False):
            images_f = images.float()

            # mu and alpha: accept 8-dim LHC vectors, adapt to K
            _mu_lhc       = kwargs.get("mu_lhc",       None)
            _alpha_lhc_raw = kwargs.get("alpha_lhc_raw", None)

            def _adapt(src: torch.Tensor, K: int) -> torch.Tensor:
                """Slice first K cols or tile 8-dim vector to length K."""
                n = src.shape[1]
                if n >= K:
                    return src[:, :K]
                repeats = (K + n - 1) // n
                return src.repeat(1, repeats)[:, :K]

            if _mu_lhc is not None:
                mu_base = _adapt(
                    torch.as_tensor(_mu_lhc, device=device, dtype=torch.float32).view(B, -1), K)
            else:
                mu_base = _shared_rand((B, K), device=device, dtype=torch.float32)

            if _alpha_lhc_raw is not None:
                alpha_base = _adapt(
                    torch.as_tensor(_alpha_lhc_raw, device=device, dtype=torch.float32).view(B, -1), K
                ) * 1.5 + 0.5
            else:
                alpha_base = _shared_rand((B, K), device=device, dtype=torch.float32) * 1.5 + 0.5

            q_edges   = torch.linspace(0, 1, K + 1, device=device)
            c_i       = torch.bucketize(images_f, q_edges) - 1
            c_i       = c_i.clamp(0, K - 1)

            mu_c    = mu_base.view(B, K, 1, 1, 1).expand(B, K, D, H, W).gather(1, c_i)
            alpha_c = alpha_base.view(B, K, 1, 1, 1).expand(B, K, D, H, W).gather(1, c_i)
            q_lower = q_edges[:-1].view(1, K, 1, 1, 1).expand(B, K, D, H, W).gather(1, c_i)
            q_upper = q_edges[1:].view(1, K, 1, 1, 1).expand(B, K, D, H, W).gather(1, c_i)
            q_center = (q_lower + q_upper) * 0.5

            y_base = mu_c + alpha_c * (images_f - q_center)
            y = torch.where(mask, y_base.to(dtype), y)

        y = torch.clamp(y, 0.0, 1.0)
        y = torch.where(mask, y, torch.zeros_like(y))

        target_hist = hist_module(y)
        return target_hist, y, y


class V24FgQuantileChunkTargetGenerator(BaseTargetGenerator):
    """
    V24: V23 with foreground-quantile chunk boundaries instead of uniform linspace.

    V23 used q_edges = linspace(0, 1, K+1), placing boundaries at uniform intensity
    thresholds. In T1w, 60-90% of foreground voxels fall below 0.5, so with K=2
    the entire "dark" chunk contains CSF + GM together — preventing the independent
    remapping required to produce T2w-like contrast (CSF bright, GM medium, WM dark).

    V24 computes q_edges from the foreground quantile distribution (voxels above
    dark_threshold) so each chunk contains an equal mass of brain tissue. With K=3
    the chunks naturally align with CSF / GM / WM, and a random draw where the CSF
    chunk gets a high target and the WM chunk a low target "accidentally" produces
    T2w-like guidance maps without any label supervision.
    """

    K_CHOICES: list[int] = [2, 3, 4, 6, 8, 12, 16]

    def __call__(
        self,
        input_images: torch.Tensor,
        hist_module: nn.Module,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = input_images
        B, C, D, H, W = images.shape
        device = images.device
        dtype = images.dtype

        dark_thr = float(kwargs.get("dark_threshold", 0.02))
        mask = images > dark_thr
        y = images.clone()

        k_idx = int(_shared_rand((1,), device=device, dtype=torch.float32).item()
                    * len(self.K_CHOICES))
        K = self.K_CHOICES[min(k_idx, len(self.K_CHOICES) - 1)]

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                            enabled=False):
            images_f = images.float()

            _mu_lhc        = kwargs.get("mu_lhc",        None)
            _alpha_lhc_raw = kwargs.get("alpha_lhc_raw", None)

            def _adapt(src: torch.Tensor, K: int) -> torch.Tensor:
                n = src.shape[1]
                if n >= K:
                    return src[:, :K]
                repeats = (K + n - 1) // n
                return src.repeat(1, repeats)[:, :K]

            if _mu_lhc is not None:
                mu_base = _adapt(
                    torch.as_tensor(_mu_lhc, device=device, dtype=torch.float32).view(B, -1), K)
            else:
                mu_base = _shared_rand((B, K), device=device, dtype=torch.float32)

            if _alpha_lhc_raw is not None:
                alpha_base = _adapt(
                    torch.as_tensor(_alpha_lhc_raw, device=device, dtype=torch.float32).view(B, -1), K
                ) * 1.5 + 0.5
            else:
                alpha_base = _shared_rand((B, K), device=device, dtype=torch.float32) * 1.5 + 0.5

            probs = torch.linspace(0, 1, K + 1, device=device, dtype=torch.float32)

            for b in range(B):
                fg_vals = images_f[b][mask[b]]
                if fg_vals.numel() >= K + 1:
                    q_edges = torch.quantile(fg_vals, probs)
                    q_edges[0]  = 0.0
                    q_edges[-1] = 1.0
                else:
                    q_edges = probs.clone()

                c_i = (torch.bucketize(images_f[b], q_edges) - 1).clamp(0, K - 1)

                q_lower  = q_edges[:-1][c_i]
                q_upper  = q_edges[1:][c_i]
                q_center = (q_lower + q_upper) * 0.5

                mu_c    = mu_base[b][c_i]
                alpha_c = alpha_base[b][c_i]

                y_base  = mu_c + alpha_c * (images_f[b] - q_center)
                y[b]    = torch.where(mask[b], y_base.to(dtype), y[b])

        y = y.clamp(0.0, 1.0)
        y = torch.where(mask, y, torch.zeros_like(y))

        target_hist = hist_module(y)
        return target_hist, y, y


class V24NonMonotoneTargetGenerator(BaseTargetGenerator):
    """
    v24_nm: Non-monotone 4-knot piecewise linear mapping.

    Insight: chunk-based generators (V24, V24Pdw) fail because foreground-quantile
    chunk boundaries don't align with SynthSeg anatomical regions — many SynthSeg WM
    labels fall in the "CSF chunk" (low-intensity partial volumes), getting mapped to
    mu_csf ≈ 0, driving WM histogram to bin 0.

    This generator bypasses chunk boundaries entirely and uses a 4-knot piecewise
    linear function on raw voxel intensity with a deliberate non-monotone segment:

      y=0 ──── x1 ────rising──── x2 ─falling─ x3 ─rising──── x=1.0, y=1.0
                              (GM peak)      (WM min)

    The V-shape minimum at (x3, y_WM) becomes the WM histogram mode.  Brightest WM
    voxels (x → 1.0) output 1.0 and naturally set p99 ≈ 1.0 in the feature extractor,
    so bins are directly y × 63:  y_GM→bin45, y_WM→bin36 match ON-Harmony T2w cluster.

    All breakpoints (x1,x2,x3) and control values (y_GM, y_WM) are sampled randomly
    within tissue-motivated ranges so the mapping varies across volumes/variants.
    """

    def __call__(
        self,
        input_images: torch.Tensor,
        hist_module: nn.Module,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = input_images
        B, C, D, H, W = images.shape
        device = images.device
        dtype = images.dtype

        dark_thr = float(kwargs.get("dark_threshold", 0.02))
        mask = images > dark_thr
        y_out = torch.zeros_like(images)

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=False):
            images_f = images.float()
            rng = lambda lo, hi: torch.rand(1, device=device).item() * (hi - lo) + lo

            for b in range(B):
                x1 = rng(0.05, 0.15)   # dark→CSF/GM transition
                x2 = rng(0.30, 0.50)   # GM→WM transition (output peaks here)
                x3 = rng(0.60, 0.80)   # WM min (V-shape bottom, mode of WM histogram)
                if x2 >= x3:           # keep ordering
                    x2, x3 = x3 - 0.05, x3

                y_GM  = rng(0.65, 0.80)   # output at x2 (GM brightness peak)
                y_WM  = rng(0.45, 0.62)   # output at x3 (WM histogram mode)
                # y=0 at x≤x1, y=1.0 at x=1.0 (sets p99 in feature extractor)

                xi = images_f[b]
                yi = torch.zeros_like(xi)

                # segment [x1, x2]: rising ramp 0 → y_GM
                seg12 = (xi >= x1) & (xi < x2)
                if seg12.any():
                    yi[seg12] = (xi[seg12] - x1) / (x2 - x1) * y_GM

                # segment [x2, x3]: falling ramp y_GM → y_WM  (non-monotone!)
                seg23 = (xi >= x2) & (xi < x3)
                if seg23.any():
                    t = (xi[seg23] - x2) / (x3 - x2)
                    yi[seg23] = y_GM + t * (y_WM - y_GM)

                # segment [x3, 1.0]: rising ramp y_WM → 1.0  (bright WM sets p99)
                seg34 = xi >= x3
                if seg34.any():
                    t = (xi[seg34] - x3) / (1.0 - x3)
                    yi[seg34] = y_WM + t * (1.0 - y_WM)

                y_out[b] = torch.where(mask[b], yi.to(dtype), torch.zeros_like(images[b]))

        y_out = y_out.clamp(0.0, 1.0)
        target_hist = hist_module(y_out)
        return target_hist, y_out, y_out


class V24NonMonotoneAdaptiveTargetGenerator(BaseTargetGenerator):
    """
    v24_nm2: Adaptive non-monotone piecewise linear mapping.

    Fixes v24_nm which used fixed absolute breakpoints (x3 ≈ 0.60-0.80) that were
    too low: T1w WM mode sits at foreground q71-89% (varies hugely by scanner) so WM
    fell in the rising segment instead of the falling one.

    Key design:
      x2 = fg.quantile(q2) where q2 ~ U(0.55, 0.75)  →  peak BELOW WM mode
      x3 = fg.quantile(q3) where q3 ~ U(0.75, 0.90)  →  V-bottom AT/ABOVE WM mode
      y_GM ~ U(0.68, 0.80)
      y_WM = R × y_GM  where R ~ U(0.45, 0.65)

    With x3 at the WM T1w mode, WM histogram peaks at output = y_WM.
    p99_brain ≈ y_GM (mapping peak), so WM bin ≈ R × 63 ∈ [28, 41] (target 36).
    """

    def __call__(
        self,
        input_images: torch.Tensor,
        hist_module: nn.Module,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = input_images
        B, C, D, H, W = images.shape
        device = images.device
        dtype = images.dtype

        dark_thr = float(kwargs.get("dark_threshold", 0.02))
        mask = images > dark_thr
        y_out = torch.zeros_like(images)

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=False):
            images_f = images.float()
            rng = lambda lo, hi: torch.rand(1, device=device).item() * (hi - lo) + lo

            for b in range(B):
                xi = images_f[b]
                fg = xi[xi > dark_thr]
                if fg.numel() < 100:
                    continue
                fg_sorted, _ = fg.flatten().sort()
                n = fg_sorted.numel()

                # Adaptive breakpoints: quantile of foreground
                q2 = rng(0.55, 0.75)
                q3 = rng(0.75, 0.90)
                if q2 >= q3:
                    q2, q3 = q3 - 0.05, q3
                x1 = dark_thr
                x2 = float(fg_sorted[min(int(q2 * n), n - 1)])
                x3 = float(fg_sorted[min(int(q3 * n), n - 1)])
                if x2 >= x3:
                    x3 = x2 + 1e-4

                y_GM = rng(0.68, 0.80)
                R    = rng(0.50, 0.70)          # y_WM / y_GM; WM bin ≈ R×63 ≈ 32–44
                y_WM = R * y_GM

                yi = torch.zeros_like(xi)

                seg12 = (xi >= x1) & (xi < x2)
                if seg12.any():
                    yi[seg12] = (xi[seg12] - x1) / (x2 - x1) * y_GM

                seg23 = (xi >= x2) & (xi < x3)
                if seg23.any():
                    t = (xi[seg23] - x2) / (x3 - x2)
                    yi[seg23] = y_GM + t * (y_WM - y_GM)

                seg34 = xi >= x3
                if seg34.any():
                    t = (xi[seg34] - x3) / max(1.0 - x3, 1e-4)
                    yi[seg34] = y_WM + t * (1.0 - y_WM)

                y_out[b] = torch.where(mask[b], yi.to(dtype), torch.zeros_like(xi))

        y_out = y_out.clamp(0.0, 1.0)
        target_hist = hist_module(y_out)
        return target_hist, y_out, y_out


class V25SpatialModulatorTargetGenerator(nn.Module):
    """
    V25: random ellipsoidal blobs, each with an independent V23RandomChunk remap.
    No labels, no atlas, no target-contrast bias.

    Base guidance map covers the whole volume.  0-3 random ellipsoidal blobs
    (hard boundaries, random foreground center, anisotropic radii 15-80 vox)
    are stamped on top, each with its own independent chunk remapping.
    Tissue boundaries within each blob are fully preserved because every map
    is intensity-based; only the regional contrast assignment changes.
    """

    MAX_BLOBS  = 3
    RADIUS_MIN = 15   # voxels (= mm at 1 mm iso)
    RADIUS_MAX = 80

    def __init__(self, **kwargs):
        super().__init__()
        self.base = V23RandomChunkTargetGenerator(**kwargs)

    def forward(
        self,
        input_images: torch.Tensor,
        num_bins: int,
        num_chunks: int,
        dark_threshold: float,
        hist_module,
        return_guidance_map: bool = False,
        labels=None,
        mu_lhc=None,
        alpha_lhc_raw=None,
    ):
        B, _, D, H, W = input_images.shape
        device = input_images.device
        fg = (input_images[:, :1] > dark_threshold).squeeze(1)  # B×D×H×W bool

        def _call_base():
            _, _, g = self.base(
                input_images=input_images, num_bins=num_bins, num_chunks=num_chunks,
                dark_threshold=dark_threshold, hist_module=hist_module,
                return_guidance_map=True, labels=None,
                mu_lhc=mu_lhc, alpha_lhc_raw=alpha_lhc_raw,
            )
            return g.float().squeeze(1)  # B×D×H×W

        out = _call_base()  # base guidance map for every batch item

        # Draw per-item blob count
        n_blobs = torch.randint(0, self.MAX_BLOBS + 1, (B,))
        max_blobs = int(n_blobs.max().item())

        if max_blobs > 0:
            # Coordinate grids — built once, shared across blobs
            zz, yy, xx = torch.meshgrid(
                torch.arange(D, device=device, dtype=torch.float32),
                torch.arange(H, device=device, dtype=torch.float32),
                torch.arange(W, device=device, dtype=torch.float32),
                indexing="ij",
            )

            # One fresh remapping per blob slot (each gets its own K/mu/alpha draw)
            blob_maps = [_call_base() for _ in range(max_blobs)]

            rng = torch.Generator(device="cpu")  # CPU generator avoids CUDA sync
            for b in range(B):
                fg_coords = fg[b].nonzero()  # N×3, dtype=long
                if fg_coords.numel() == 0:
                    continue
                for slot in range(int(n_blobs[b].item())):
                    # Random center inside foreground
                    ci = torch.randint(0, fg_coords.shape[0], (1,), generator=rng).item()
                    cz, cy, cx = fg_coords[ci].float()

                    # Anisotropic radii
                    rz = torch.empty(1).uniform_(self.RADIUS_MIN, self.RADIUS_MAX).item()
                    ry = torch.empty(1).uniform_(self.RADIUS_MIN, self.RADIUS_MAX).item()
                    rx = torch.empty(1).uniform_(self.RADIUS_MIN, self.RADIUS_MAX).item()

                    mask = (
                        ((zz - cz) / rz) ** 2
                        + ((yy - cy) / ry) ** 2
                        + ((xx - cx) / rx) ** 2
                    ) <= 1.0
                    mask = mask & fg[b]

                    if mask.any():
                        out[b][mask] = blob_maps[slot][b][mask]

        out = out.clamp(0.0, 1.0)
        out = torch.where(fg, out, torch.zeros_like(out))
        out = out.unsqueeze(1)  # B×1×D×H×W

        target_hist = hist_module(out)
        return target_hist, out, out


# SynthSeg FreeSurfer label sets for each tissue class
_LABEL_WM_SET          = frozenset([2, 41])
_LABEL_CORTEX_SET      = frozenset([3, 42])
_LABEL_CSF_SET         = frozenset([4, 5, 14, 15, 43, 44])
_LABEL_CEREB_SET       = frozenset([7, 8, 46, 47])
_LABEL_BRAINSTEM_SET   = frozenset([16])
_LABEL_THALAMUS_SET    = frozenset([10, 49])
_LABEL_STRIATUM_SET    = frozenset([11, 12, 50, 51])
_LABEL_PALLIDUM_SET    = frozenset([13, 52])
_LABEL_HIPPOAMYG_SET   = frozenset([17, 18, 53, 54])
_LABEL_OTHERSUB_SET    = frozenset([26, 28, 58, 60])


class V24T2wTargetGenerator(BaseTargetGenerator):
    """
    v24_t2w: K=3 with T2w-biased mu ranges per chunk.

    Foreground-quantile boundaries (same as V24) align chunk 0 with CSF, 1 with GM,
    2 with WM. Instead of uniform mu, each chunk samples from a range that biases
    toward the T2w tissue ordering (CSF bright, GM medium, WM dark):
      chunk 0 (CSF): mu ~ U(0.6, 1.0)
      chunk 1 (GM):  mu ~ U(0.2, 0.8)
      chunk 2 (WM):  mu ~ U(0.0, 0.4)
    Alpha stays random [0.5, 2.0] for within-chunk texture diversity.
    """

    MU_RANGES = [(0.6, 1.0), (0.2, 0.8), (0.0, 0.4)]

    def __call__(
        self,
        input_images: torch.Tensor,
        hist_module: nn.Module,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = input_images
        B, C, D, H, W = images.shape
        device = images.device
        dtype = images.dtype
        K = 3

        dark_thr = float(kwargs.get("dark_threshold", 0.02))
        mask = images > dark_thr
        y = images.clone()

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=False):
            images_f = images.float()

            mu_base = torch.zeros(B, K, device=device, dtype=torch.float32)
            for k, (lo, hi) in enumerate(self.MU_RANGES):
                mu_base[:, k] = torch.rand(B, device=device) * (hi - lo) + lo

            alpha_base = torch.rand(B, K, device=device, dtype=torch.float32) * 1.5 + 0.5

            probs = torch.linspace(0, 1, K + 1, device=device, dtype=torch.float32)

            for b in range(B):
                fg_vals = images_f[b][mask[b]]
                if fg_vals.numel() >= K + 1:
                    q_edges = torch.quantile(fg_vals, probs)
                    q_edges[0]  = 0.0
                    q_edges[-1] = 1.0
                else:
                    q_edges = probs.clone()

                c_i = (torch.bucketize(images_f[b], q_edges) - 1).clamp(0, K - 1)
                q_lower  = q_edges[:-1][c_i]
                q_upper  = q_edges[1:][c_i]
                q_center = (q_lower + q_upper) * 0.5

                mu_c    = mu_base[b][c_i]
                alpha_c = alpha_base[b][c_i]

                y_base  = mu_c + alpha_c * (images_f[b] - q_center)
                y[b]    = torch.where(mask[b], y_base.to(dtype), y[b])

        y = y.clamp(0.0, 1.0)
        y = torch.where(mask, y, torch.zeros_like(y))

        target_hist = hist_module(y)
        return target_hist, y, y


class V24DescTargetGenerator(BaseTargetGenerator):
    """
    v24_desc: V24 with K∈{2,3,4} and mu sorted in descending order after sampling.

    Sorting mu descending guarantees that chunk 0 (darkest T1w tissue = CSF) always
    maps to the highest target intensity, and chunk K-1 (brightest = WM) always maps
    to the lowest — a guaranteed T2w-like inversion at every chunk boundary.
    Alpha stays random for within-chunk diversity. Smaller K pool (2,3,4) keeps the
    inversion meaningful (large K makes each chunk too narrow to represent a tissue class).
    """

    K_CHOICES: list[int] = [2, 3, 4]

    def __call__(
        self,
        input_images: torch.Tensor,
        hist_module: nn.Module,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = input_images
        B, C, D, H, W = images.shape
        device = images.device
        dtype = images.dtype

        dark_thr = float(kwargs.get("dark_threshold", 0.02))
        mask = images > dark_thr
        y = images.clone()

        k_idx = int(_shared_rand((1,), device=device, dtype=torch.float32).item()
                    * len(self.K_CHOICES))
        K = self.K_CHOICES[min(k_idx, len(self.K_CHOICES) - 1)]

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=False):
            images_f = images.float()

            mu_base = _shared_rand((B, K), device=device, dtype=torch.float32)
            mu_base, _ = mu_base.sort(dim=1, descending=True)  # chunk 0=CSF→highest target

            alpha_base = _shared_rand((B, K), device=device, dtype=torch.float32) * 1.5 + 0.5

            probs = torch.linspace(0, 1, K + 1, device=device, dtype=torch.float32)

            for b in range(B):
                fg_vals = images_f[b][mask[b]]
                if fg_vals.numel() >= K + 1:
                    q_edges = torch.quantile(fg_vals, probs)
                    q_edges[0]  = 0.0
                    q_edges[-1] = 1.0
                else:
                    q_edges = probs.clone()

                c_i = (torch.bucketize(images_f[b], q_edges) - 1).clamp(0, K - 1)
                q_lower  = q_edges[:-1][c_i]
                q_upper  = q_edges[1:][c_i]
                q_center = (q_lower + q_upper) * 0.5

                mu_c    = mu_base[b][c_i]
                alpha_c = alpha_base[b][c_i]

                y_base  = mu_c + alpha_c * (images_f[b] - q_center)
                y[b]    = torch.where(mask[b], y_base.to(dtype), y[b])

        y = y.clamp(0.0, 1.0)
        y = torch.where(mask, y, torch.zeros_like(y))

        target_hist = hist_module(y)
        return target_hist, y, y


class V24InvTargetGenerator(BaseTargetGenerator):
    """
    v24_inv: 50% global inversion (y = 1-x in foreground), 50% standard V24.

    Global inversion directly maps T1w → T2w-like: CSF (0.1) → 0.9 (bright),
    WM (0.8) → 0.2 (dark). The other 50% keep V24 diversity.
    Inversion decision is per-sample (not shared across DDP ranks).
    """

    K_CHOICES: list[int] = [2, 3, 4, 6, 8, 12, 16]

    def __call__(
        self,
        input_images: torch.Tensor,
        hist_module: nn.Module,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = input_images
        B, C, D, H, W = images.shape
        device = images.device
        dtype = images.dtype

        dark_thr = float(kwargs.get("dark_threshold", 0.02))
        mask = images > dark_thr

        k_idx = int(_shared_rand((1,), device=device, dtype=torch.float32).item()
                    * len(self.K_CHOICES))
        K = self.K_CHOICES[min(k_idx, len(self.K_CHOICES) - 1)]

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=False):
            images_f = images.float()

            mu_base = _shared_rand((B, K), device=device, dtype=torch.float32)
            alpha_base = _shared_rand((B, K), device=device, dtype=torch.float32) * 1.5 + 0.5

            probs = torch.linspace(0, 1, K + 1, device=device, dtype=torch.float32)

            y = images.clone()
            for b in range(B):
                # 50% chance: global inversion
                if torch.rand(1, device=device).item() > 0.5:
                    inv = (1.0 - images_f[b]).to(dtype)
                    y[b] = torch.where(mask[b], inv, torch.zeros_like(images[b]))
                    continue

                # Standard V24 piecewise remap
                fg_vals = images_f[b][mask[b]]
                if fg_vals.numel() >= K + 1:
                    q_edges = torch.quantile(fg_vals, probs)
                    q_edges[0]  = 0.0
                    q_edges[-1] = 1.0
                else:
                    q_edges = probs.clone()

                c_i = (torch.bucketize(images_f[b], q_edges) - 1).clamp(0, K - 1)
                q_lower  = q_edges[:-1][c_i]
                q_upper  = q_edges[1:][c_i]
                q_center = (q_lower + q_upper) * 0.5

                mu_c    = mu_base[b][c_i]
                alpha_c = alpha_base[b][c_i]

                y_base  = mu_c + alpha_c * (images_f[b] - q_center)
                y[b]    = torch.where(mask[b], y_base.to(dtype), y[b])

        y = y.clamp(0.0, 1.0)
        y = torch.where(mask, y, torch.zeros_like(y))

        target_hist = hist_module(y)
        return target_hist, y, y


class V24PdwTargetGenerator(BaseTargetGenerator):
    """
    v24_pdw: K=3, CSF dark, GM > WM. Matches ON-Harmony T2w feature cluster.

    Analysis of ON-Harmony T2w in p1-p99 normalized regional_hist_64 space shows the
    cluster is PDw-like (NOT classic T2w): CSF dark (bin 0), WM medium (bin 36), GM
    slightly brighter than WM (bin 45). v24_t2w produced the opposite (CSF bright, WM
    dark) and landed far from the real cluster.

    Targeted mu values derived from the required p1-p99 mapping:
      chunk 0 (CSF, darkest 1/3): mu ~ U(0.02, 0.08)  → stays dark, sets p1≈0
      chunk 1 (GM, middle 1/3):   mu ~ U(0.55, 0.65)  → medium-high
      chunk 2 (WM, brightest 1/3): mu ~ U(0.40, 0.50) → medium, LESS than GM

    With alpha≈1.5, bright GM voxels (x near q2=0.65) map to ~0.84, setting p99≈0.84.
    After p1-p99 normalization: WM_mode/0.84≈0.54→bin34, GM_mode/0.84≈0.71→bin45.
    """

    MU_RANGES  = [(0.02, 0.08), (0.55, 0.65), (0.40, 0.50)]  # CSF, GM, WM
    ALPHA_RANGE = (1.2, 1.8)

    def __call__(
        self,
        input_images: torch.Tensor,
        hist_module: nn.Module,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = input_images
        B, C, D, H, W = images.shape
        device = images.device
        dtype = images.dtype
        K = 3

        dark_thr = float(kwargs.get("dark_threshold", 0.02))
        mask = images > dark_thr
        y = images.clone()

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=False):
            images_f = images.float()

            mu_base = torch.zeros(B, K, device=device, dtype=torch.float32)
            for k, (lo, hi) in enumerate(self.MU_RANGES):
                mu_base[:, k] = torch.rand(B, device=device) * (hi - lo) + lo

            a_lo, a_hi = self.ALPHA_RANGE
            alpha_base = torch.rand(B, K, device=device, dtype=torch.float32) * (a_hi - a_lo) + a_lo

            probs = torch.linspace(0, 1, K + 1, device=device, dtype=torch.float32)

            for b in range(B):
                fg_vals = images_f[b][mask[b]]
                if fg_vals.numel() >= K + 1:
                    q_edges = torch.quantile(fg_vals, probs)
                    q_edges[0]  = 0.0
                    q_edges[-1] = 1.0
                else:
                    q_edges = probs.clone()

                c_i = (torch.bucketize(images_f[b], q_edges) - 1).clamp(0, K - 1)
                q_lower  = q_edges[:-1][c_i]
                q_upper  = q_edges[1:][c_i]
                q_center = (q_lower + q_upper) * 0.5

                mu_c    = mu_base[b][c_i]
                alpha_c = alpha_base[b][c_i]

                y_base  = mu_c + alpha_c * (images_f[b] - q_center)
                y[b]    = torch.where(mask[b], y_base.to(dtype), y[b])

        y = y.clamp(0.0, 1.0)
        y = torch.where(mask, y, torch.zeros_like(y))

        target_hist = hist_module(y)
        return target_hist, y, y


class SynthSegBaselineTargetGenerator(BaseTargetGenerator):
    """
    SynthSeg Baseline: Applies pure SynthSeg GMM sampling exclusively to the
    available tumor labels (NCR, ED, ET handled separately) while leaving the
    background intensities unaugmented, serving as a direct ablation study 
    for partial-label generalization scenarios.
    """
    def __call__(
        self,
        input_images: torch.Tensor,
        hist_module: nn.Module,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = input_images
        B, C, D, H, W = images.shape
        device = images.device
        dtype = images.dtype

        # Step A: Initialize
        y = images.clone()
        mask = images > 0.01

        # Step B: Pure GMM Sampling on Masks
        if labels is not None:
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=False):
                y_f = y.float()
                
                for c in [1, 2, 3]:
                    class_mask = (labels == c)
                    
                    mu_c = _shared_rand((B, 1, 1, 1, 1), device=device, dtype=torch.float32)
                    sigma_c = _shared_rand((B, 1, 1, 1, 1), device=device, dtype=torch.float32) * 0.09 + 0.01
                    
                    Z = torch.randn_like(y_f)
                    y_synthseg = mu_c + sigma_c * Z
                    
                    y_f = torch.where(class_mask, y_synthseg, y_f)
                    
                y = y_f.to(dtype)
                
        # Step C: Normalize & Mask (v26_4+: min-max within mask; revert: torch.clamp(y,0,1))
        y = _normalize_guidance(y.float(), mask.float()).to(dtype)
        y = torch.where(mask, y, torch.zeros_like(y))

        target_hist = hist_module(y)
        return target_hist, y, y


class V26EMParcellationChunkTargetGenerator(BaseTargetGenerator):
    """
    V26_1: Unsupervised EM-like Spatial Parcellation.
    1. Extracts C random intensity classes (1D K-means) from T1w foreground.
    2. Spatially subdivides each class mask into Sc subregions (Random Voronoi).
    3. Remaps each subregion independently (flat affine: mu + alpha*img).

    Fully label-free and unsupervised, achieving spatial intensity decoupling
    without anatomical priors.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.C_CHOICES = [2, 3, 4, 5, 6]
        self.S_CHOICES = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    def __call__(
        self,
        input_images: torch.Tensor,
        hist_module: nn.Module,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = input_images
        B, C_ch, D, H, W = images.shape
        device = images.device
        dtype = images.dtype
        mask = images > 0.01

        y = images.clone()

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=False):
            images_f = images.float()

            for b in range(B):
                b_mask = mask[b, 0]
                if not b_mask.any():
                    continue

                img_b = images_f[b, 0]
                fg_vals = img_b[b_mask]
                
                # 1. 1D K-means for intensity classes
                c_idx = int(_shared_rand((1,), device=device, dtype=torch.float32).item() * len(self.C_CHOICES))
                C = self.C_CHOICES[min(c_idx, len(self.C_CHOICES) - 1)]
                
                N_fg = fg_vals.shape[0]
                n_samples = min(N_fg, 10000)
                sub_fg = fg_vals[torch.randperm(N_fg, device=device)[:n_samples]]
                
                centroids = torch.linspace(sub_fg.min(), sub_fg.max(), C, device=device)
                for _ in range(10):
                    dists = torch.abs(sub_fg.unsqueeze(1) - centroids.unsqueeze(0))
                    lbls = torch.argmin(dists, dim=1)
                    sums = torch.zeros(C, device=device).scatter_add_(0, lbls, sub_fg)
                    counts = torch.zeros(C, device=device).scatter_add_(0, lbls, torch.ones_like(sub_fg))
                    new_centroids = torch.where(counts > 0, sums / counts, centroids)
                    if torch.allclose(centroids, new_centroids):
                        break
                    centroids = new_centroids
                
                dists_all = torch.abs(img_b.unsqueeze(-1) - centroids.view(1, 1, 1, C))
                class_lbls = torch.argmin(dists_all, dim=-1)
                
                y_b = y[b, 0]
                coords_z, coords_y, coords_x = torch.meshgrid(
                    torch.arange(D, device=device),
                    torch.arange(H, device=device),
                    torch.arange(W, device=device),
                    indexing="ij"
                )
                coords = torch.stack([coords_z, coords_y, coords_x], dim=-1).float()
                
                for c in range(C):
                    c_mask = b_mask & (class_lbls == c)
                    N_c = c_mask.sum()
                    if N_c == 0:
                        continue
                        
                    s_idx = int(torch.rand((1,), device=device).item() * len(self.S_CHOICES))
                    S = self.S_CHOICES[min(s_idx, len(self.S_CHOICES)-1)]
                    
                    S = min(S, N_c.item())
                    if S == 0:
                        continue
                        
                    c_coords = coords[c_mask]
                    c_centroids = c_coords[torch.randperm(N_c, device=device)[:S]]
                    c_dists = torch.cdist(c_coords, c_centroids)
                    sub_lbls = torch.argmin(c_dists, dim=1)
                    
                    c_mask_flat = c_mask.view(-1)
                    sub_lbls_flat = torch.full_like(c_mask_flat, -1, dtype=torch.long)
                    sub_lbls_flat[c_mask_flat] = sub_lbls
                    sub_lbls_3d = sub_lbls_flat.view(D, H, W)
                    
                    for s in range(S):
                        s_mask_global = c_mask & (sub_lbls_3d == s)
                        if not s_mask_global.any():
                            continue
                            
                        # Use same random draw ranges as V23
                        mu = torch.rand((1,), device=device).item()
                        alpha = torch.rand((1,), device=device).item() * 1.5 + 0.5
                        
                        q_center = img_b[s_mask_global].mean()
                        remap = mu + alpha * (img_b[s_mask_global] - q_center)
                        
                        y_b[s_mask_global] = remap.to(dtype)

        return y, y, y


class V26_2EMParcellationTargetGenerator(BaseTargetGenerator):
    """
    V26_2: Improved Unsupervised EM-like Spatial Parcellation.
    1. 30% chance to skip parcellation entirely.
    2. When parcellating with 1D K-means, each class has a 40% chance to NOT be split in subregions.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.C_CHOICES = [2, 3, 4, 5, 6]
        self.S_CHOICES = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    def __call__(
        self,
        input_images: torch.Tensor,
        hist_module: nn.Module,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = input_images
        B, C_ch, D, H, W = images.shape
        device = images.device
        dtype = images.dtype
        mask = images > 0.01

        y = images.clone()

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=False):
            images_f = images.float()

            for b in range(B):
                b_mask = mask[b, 0]
                if not b_mask.any():
                    continue

                img_b = images_f[b, 0]
                fg_vals = img_b[b_mask]
                
                # 30% chance to not parcellate at all
                skip_parcellation = (torch.rand((1,), device=device).item() < 0.3)
                
                y_b = y[b, 0]
                
                if skip_parcellation:
                    mu = torch.rand((1,), device=device).item()
                    alpha = torch.rand((1,), device=device).item() * 1.5 + 0.5
                    q_center = fg_vals.mean()
                    remap = mu + alpha * (img_b[b_mask] - q_center)
                    y_b[b_mask] = remap.to(dtype)
                    continue

                # 1. 1D K-means for intensity classes
                c_idx = int(torch.rand((1,), device=device).item() * len(self.C_CHOICES))
                C = self.C_CHOICES[min(c_idx, len(self.C_CHOICES) - 1)]
                
                N_fg = fg_vals.shape[0]
                n_samples = min(N_fg, 10000)
                sub_fg = fg_vals[torch.randperm(N_fg, device=device)[:n_samples]]
                
                centroids = torch.linspace(sub_fg.min(), sub_fg.max(), C, device=device)
                for _ in range(10):
                    dists = torch.abs(sub_fg.unsqueeze(1) - centroids.unsqueeze(0))
                    lbls = torch.argmin(dists, dim=1)
                    sums = torch.zeros(C, device=device).scatter_add_(0, lbls, sub_fg)
                    counts = torch.zeros(C, device=device).scatter_add_(0, lbls, torch.ones_like(sub_fg))
                    new_centroids = torch.where(counts > 0, sums / counts, centroids)
                    if torch.allclose(centroids, new_centroids):
                        break
                    centroids = new_centroids
                
                dists_all = torch.abs(img_b.unsqueeze(-1) - centroids.view(1, 1, 1, C))
                class_lbls = torch.argmin(dists_all, dim=-1)
                
                coords_z, coords_y, coords_x = torch.meshgrid(
                    torch.arange(D, device=device),
                    torch.arange(H, device=device),
                    torch.arange(W, device=device),
                    indexing="ij"
                )
                coords = torch.stack([coords_z, coords_y, coords_x], dim=-1).float()
                
                for c in range(C):
                    c_mask = b_mask & (class_lbls == c)
                    N_c = c_mask.sum()
                    if N_c == 0:
                        continue
                        
                    skip_sub_parc = (torch.rand((1,), device=device).item() < 0.4)
                    
                    if skip_sub_parc:
                        S = 1
                    else:
                        s_idx = int(torch.rand((1,), device=device).item() * len(self.S_CHOICES))
                        S = self.S_CHOICES[min(s_idx, len(self.S_CHOICES)-1)]
                    
                    S = min(S, N_c.item())
                    if S == 0:
                        continue
                        
                    if S == 1:
                        s_mask_global = c_mask
                        mu = torch.rand((1,), device=device).item()
                        alpha = torch.rand((1,), device=device).item() * 1.5 + 0.5
                        q_center = img_b[s_mask_global].mean()
                        remap = mu + alpha * (img_b[s_mask_global] - q_center)
                        y_b[s_mask_global] = remap.to(dtype)
                        continue
                        
                    c_coords = coords[c_mask]
                    c_centroids = c_coords[torch.randperm(N_c, device=device)[:S]]
                    c_dists = torch.cdist(c_coords, c_centroids)
                    sub_lbls = torch.argmin(c_dists, dim=1)
                    
                    c_mask_flat = c_mask.view(-1)
                    sub_lbls_flat = torch.full_like(c_mask_flat, -1, dtype=torch.long)
                    sub_lbls_flat[c_mask_flat] = sub_lbls
                    sub_lbls_3d = sub_lbls_flat.view(D, H, W)
                    
                    for s in range(S):
                        s_mask_global = c_mask & (sub_lbls_3d == s)
                        if not s_mask_global.any():
                            continue
                            
                        mu = torch.rand((1,), device=device).item()
                        alpha = torch.rand((1,), device=device).item() * 1.5 + 0.5
                        
                        q_center = img_b[s_mask_global].mean()
                        remap = mu + alpha * (img_b[s_mask_global] - q_center)
                        
                        y_b[s_mask_global] = remap.to(dtype)

        return y, y, y


class V26_3EMParcellationTargetGenerator(V26_2EMParcellationTargetGenerator):
    """
    V26_3: V26_2 with higher parcellation probability (10% skip vs 30%).
    Sub-parcellation skip unchanged at 40%.
    """
    SKIP_PARCELLATION_PROB = 0.10
    SKIP_SUB_PARC_PROB     = 0.40

    def __call__(self, input_images, hist_module, labels=None, **kwargs):
        images = input_images
        B, C_ch, D, H, W = images.shape
        device = images.device
        dtype  = images.dtype
        mask   = images > 0.01
        y      = images.clone()

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=False):
            images_f = images.float()
            for b in range(B):
                b_mask = mask[b, 0]
                if not b_mask.any():
                    continue
                img_b   = images_f[b, 0]
                fg_vals = img_b[b_mask]
                y_b     = y[b, 0]

                if torch.rand((1,), device=device).item() < self.SKIP_PARCELLATION_PROB:
                    mu    = torch.rand((1,), device=device).item()
                    alpha = torch.rand((1,), device=device).item() * 1.5 + 0.5
                    remap = mu + alpha * (img_b[b_mask] - fg_vals.mean())
                    y_b[b_mask] = remap.to(dtype)
                    continue

                c_idx     = int(torch.rand((1,), device=device).item() * len(self.C_CHOICES))
                C         = self.C_CHOICES[min(c_idx, len(self.C_CHOICES) - 1)]
                N_fg      = fg_vals.shape[0]
                sub_fg    = fg_vals[torch.randperm(N_fg, device=device)[:min(N_fg, 10000)]]
                centroids = torch.linspace(sub_fg.min(), sub_fg.max(), C, device=device)
                for _ in range(10):
                    dists = torch.abs(sub_fg.unsqueeze(1) - centroids.unsqueeze(0))
                    lbls  = torch.argmin(dists, dim=1)
                    sums  = torch.zeros(C, device=device).scatter_add_(0, lbls, sub_fg)
                    cnts  = torch.zeros(C, device=device).scatter_add_(0, lbls, torch.ones_like(sub_fg))
                    new_c = torch.where(cnts > 0, sums / cnts, centroids)
                    if torch.allclose(centroids, new_c): break
                    centroids = new_c

                dists_all  = torch.abs(img_b.unsqueeze(-1) - centroids.view(1, 1, 1, C))
                class_lbls = torch.argmin(dists_all, dim=-1)
                coords     = torch.stack(torch.meshgrid(
                    torch.arange(D, device=device),
                    torch.arange(H, device=device),
                    torch.arange(W, device=device),
                    indexing="ij"), dim=-1).float()

                for c in range(C):
                    c_mask = b_mask & (class_lbls == c)
                    N_c    = c_mask.sum()
                    if N_c == 0:
                        continue
                    if torch.rand((1,), device=device).item() < self.SKIP_SUB_PARC_PROB:
                        S = 1
                    else:
                        s_idx = int(torch.rand((1,), device=device).item() * len(self.S_CHOICES))
                        S = min(self.S_CHOICES[min(s_idx, len(self.S_CHOICES)-1)], N_c.item())
                    if S == 0:
                        continue
                    if S == 1:
                        mu    = torch.rand((1,), device=device).item()
                        alpha = torch.rand((1,), device=device).item() * 1.5 + 0.5
                        remap = mu + alpha * (img_b[c_mask] - img_b[c_mask].mean())
                        y_b[c_mask] = remap.to(dtype)
                        continue
                    c_coords    = coords[c_mask]
                    c_centroids = c_coords[torch.randperm(N_c, device=device)[:S]]
                    sub_lbls    = torch.argmin(torch.cdist(c_coords, c_centroids), dim=1)
                    sl_flat     = torch.full((D*H*W,), -1, dtype=torch.long, device=device)
                    sl_flat[c_mask.view(-1)] = sub_lbls
                    sub_lbls_3d = sl_flat.view(D, H, W)
                    for s in range(S):
                        s_mask = c_mask & (sub_lbls_3d == s)
                        if not s_mask.any():
                            continue
                        mu    = torch.rand((1,), device=device).item()
                        alpha = torch.rand((1,), device=device).item() * 1.5 + 0.5
                        remap = mu + alpha * (img_b[s_mask] - img_b[s_mask].mean())
                        y_b[s_mask] = remap.to(dtype)

        # Normalize & Mask (v26_4+: min-max within mask; revert: torch.clamp(y,0,1))
        y = _normalize_guidance(y.float(), mask.float()).to(dtype)
        y = torch.where(mask, y, torch.zeros_like(y))
        return y, y, y


class V26_4EMParcellationTargetGenerator(V26_3EMParcellationTargetGenerator):
    """V26_4: V26_3 with per-sample min-max normalization replacing hard clamp.
    Preserves full dynamic range of affine remapping — no saturation at 0/1.
    Identical to V26_3 in all other respects (blur list, skip probs, parcellation)."""
    pass


class V26_5PolarizedTargetGenerator(V26_4EMParcellationTargetGenerator):
    """V26_5: V26_4 with coherent cross-class contrast polarity.

    50% of images: intensity classes receive target mu values in DESCENDING order
    of their T1w centroid (darkest T1w tissue → highest output brightness).
    This is "anti-T1w" polarity — directly produces T2w/FLAIR-like global contrast.

    50% of images: mu values drawn independently per sub-region (V26_4 behaviour).

    Within each class, all sub-regions share the class-level mu; only alpha varies
    spatially. This ensures coherent tissue-level contrast while preserving local
    intensity texture variation.
    """

    def __call__(self, input_images, hist_module, labels=None, **kwargs):
        images = input_images
        B, C_ch, D, H, W = images.shape
        device = images.device
        dtype  = images.dtype
        mask   = images > 0.01
        y      = images.clone()

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=False):
            images_f = images.float()
            for b in range(B):
                b_mask = mask[b, 0]
                if not b_mask.any():
                    continue
                img_b   = images_f[b, 0]
                fg_vals = img_b[b_mask]
                y_b     = y[b, 0]

                if torch.rand((1,), device=device).item() < self.SKIP_PARCELLATION_PROB:
                    mu    = torch.rand((1,), device=device).item()
                    alpha = torch.rand((1,), device=device).item() * 1.5 + 0.5
                    y_b[b_mask] = (mu + alpha * (img_b[b_mask] - fg_vals.mean())).to(dtype)
                    continue

                c_idx     = int(torch.rand((1,), device=device).item() * len(self.C_CHOICES))
                C         = self.C_CHOICES[min(c_idx, len(self.C_CHOICES) - 1)]
                N_fg      = fg_vals.shape[0]
                sub_fg    = fg_vals[torch.randperm(N_fg, device=device)[:min(N_fg, 10000)]]
                centroids = torch.linspace(sub_fg.min(), sub_fg.max(), C, device=device)
                for _ in range(10):
                    dists = torch.abs(sub_fg.unsqueeze(1) - centroids.unsqueeze(0))
                    lbls  = torch.argmin(dists, dim=1)
                    sums  = torch.zeros(C, device=device).scatter_add_(0, lbls, sub_fg)
                    cnts  = torch.zeros(C, device=device).scatter_add_(0, lbls, torch.ones_like(sub_fg))
                    new_c = torch.where(cnts > 0, sums / cnts, centroids)
                    if torch.allclose(centroids, new_c): break
                    centroids = new_c

                dists_all  = torch.abs(img_b.unsqueeze(-1) - centroids.view(1, 1, 1, C))
                class_lbls = torch.argmin(dists_all, dim=-1)
                coords     = torch.stack(torch.meshgrid(
                    torch.arange(D, device=device),
                    torch.arange(H, device=device),
                    torch.arange(W, device=device),
                    indexing="ij"), dim=-1).float()

                # Polarity: sort class mus inversely w.r.t. T1w centroid ordering.
                # sort_idx[rank] = class_id → mu_per_class[class_id] = draw[rank]
                sort_idx       = torch.argsort(centroids)          # ascending T1w order
                mu_draws       = torch.rand(C, device=device)
                invert         = torch.rand((1,), device=device).item() < 0.5
                if invert:
                    mu_draws, _ = torch.sort(mu_draws, descending=True)  # rank 0 (darkest) → highest mu
                mu_per_class   = torch.zeros(C, device=device)
                mu_per_class[sort_idx] = mu_draws

                for c in range(C):
                    c_mask   = b_mask & (class_lbls == c)
                    N_c      = c_mask.sum()
                    if N_c == 0:
                        continue
                    class_mu = mu_per_class[c].item()
                    if torch.rand((1,), device=device).item() < self.SKIP_SUB_PARC_PROB:
                        S = 1
                    else:
                        s_idx = int(torch.rand((1,), device=device).item() * len(self.S_CHOICES))
                        S = min(self.S_CHOICES[min(s_idx, len(self.S_CHOICES) - 1)], N_c.item())
                    if S == 0:
                        continue
                    if S == 1:
                        alpha = torch.rand((1,), device=device).item() * 1.5 + 0.5
                        y_b[c_mask] = (class_mu + alpha * (img_b[c_mask] - img_b[c_mask].mean())).to(dtype)
                        continue
                    c_coords    = coords[c_mask]
                    c_centroids = c_coords[torch.randperm(N_c, device=device)[:S]]
                    sub_lbls    = torch.argmin(torch.cdist(c_coords, c_centroids), dim=1)
                    sl_flat     = torch.full((D * H * W,), -1, dtype=torch.long, device=device)
                    sl_flat[c_mask.view(-1)] = sub_lbls
                    sub_lbls_3d = sl_flat.view(D, H, W)
                    for s in range(S):
                        s_mask = c_mask & (sub_lbls_3d == s)
                        if not s_mask.any():
                            continue
                        alpha = torch.rand((1,), device=device).item() * 1.5 + 0.5
                        y_b[s_mask] = (class_mu + alpha * (img_b[s_mask] - img_b[s_mask].mean())).to(dtype)

        y = _normalize_guidance(y.float(), mask.float()).to(dtype)
        y = torch.where(mask, y, torch.zeros_like(y))
        return y, y, y


class V26_6SignedAlphaTargetGenerator(V26_4EMParcellationTargetGenerator):
    """V26_6: V26_4 with signed alpha (allows within-region intensity inversion).

    Each region (skip-parc fallback, per-class, or per-sub-region) draws alpha from
    a signed distribution: sign ~ Bernoulli(0.5), magnitude ~ U(0.5, 2.0).
    Negative alpha inverts the local intensity ordering within a region, creating
    contrast inversions at the sub-tissue level without changing the K-means structure.
    """

    def __call__(self, input_images, hist_module, labels=None, **kwargs):
        images = input_images
        B, C_ch, D, H, W = images.shape
        device = images.device
        dtype  = images.dtype
        mask   = images > 0.01
        y      = images.clone()

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=False):
            images_f = images.float()
            for b in range(B):
                b_mask = mask[b, 0]
                if not b_mask.any():
                    continue
                img_b   = images_f[b, 0]
                fg_vals = img_b[b_mask]
                y_b     = y[b, 0]

                def _signed_alpha() -> float:
                    sign = 1.0 if torch.rand((1,), device=device).item() > 0.5 else -1.0
                    return sign * (torch.rand((1,), device=device).item() * 1.5 + 0.5)

                if torch.rand((1,), device=device).item() < self.SKIP_PARCELLATION_PROB:
                    mu    = torch.rand((1,), device=device).item()
                    alpha = _signed_alpha()
                    y_b[b_mask] = (mu + alpha * (img_b[b_mask] - fg_vals.mean())).to(dtype)
                    continue

                c_idx     = int(torch.rand((1,), device=device).item() * len(self.C_CHOICES))
                C         = self.C_CHOICES[min(c_idx, len(self.C_CHOICES) - 1)]
                N_fg      = fg_vals.shape[0]
                sub_fg    = fg_vals[torch.randperm(N_fg, device=device)[:min(N_fg, 10000)]]
                centroids = torch.linspace(sub_fg.min(), sub_fg.max(), C, device=device)
                for _ in range(10):
                    dists = torch.abs(sub_fg.unsqueeze(1) - centroids.unsqueeze(0))
                    lbls  = torch.argmin(dists, dim=1)
                    sums  = torch.zeros(C, device=device).scatter_add_(0, lbls, sub_fg)
                    cnts  = torch.zeros(C, device=device).scatter_add_(0, lbls, torch.ones_like(sub_fg))
                    new_c = torch.where(cnts > 0, sums / cnts, centroids)
                    if torch.allclose(centroids, new_c): break
                    centroids = new_c

                dists_all  = torch.abs(img_b.unsqueeze(-1) - centroids.view(1, 1, 1, C))
                class_lbls = torch.argmin(dists_all, dim=-1)
                coords     = torch.stack(torch.meshgrid(
                    torch.arange(D, device=device),
                    torch.arange(H, device=device),
                    torch.arange(W, device=device),
                    indexing="ij"), dim=-1).float()

                for c in range(C):
                    c_mask = b_mask & (class_lbls == c)
                    N_c    = c_mask.sum()
                    if N_c == 0:
                        continue
                    if torch.rand((1,), device=device).item() < self.SKIP_SUB_PARC_PROB:
                        S = 1
                    else:
                        s_idx = int(torch.rand((1,), device=device).item() * len(self.S_CHOICES))
                        S = min(self.S_CHOICES[min(s_idx, len(self.S_CHOICES) - 1)], N_c.item())
                    if S == 0:
                        continue
                    if S == 1:
                        mu    = torch.rand((1,), device=device).item()
                        alpha = _signed_alpha()
                        y_b[c_mask] = (mu + alpha * (img_b[c_mask] - img_b[c_mask].mean())).to(dtype)
                        continue
                    c_coords    = coords[c_mask]
                    c_centroids = c_coords[torch.randperm(N_c, device=device)[:S]]
                    sub_lbls    = torch.argmin(torch.cdist(c_coords, c_centroids), dim=1)
                    sl_flat     = torch.full((D * H * W,), -1, dtype=torch.long, device=device)
                    sl_flat[c_mask.view(-1)] = sub_lbls
                    sub_lbls_3d = sl_flat.view(D, H, W)
                    for s in range(S):
                        s_mask = c_mask & (sub_lbls_3d == s)
                        if not s_mask.any():
                            continue
                        mu    = torch.rand((1,), device=device).item()
                        alpha = _signed_alpha()
                        y_b[s_mask] = (mu + alpha * (img_b[s_mask] - img_b[s_mask].mean())).to(dtype)

        y = _normalize_guidance(y.float(), mask.float()).to(dtype)
        y = torch.where(mask, y, torch.zeros_like(y))
        return y, y, y


class V26_7FlatRegionTargetGenerator(V26_4EMParcellationTargetGenerator):
    """V26_7: V26_4 parcellation with constant (flat) intensity per region.

    Replaces the affine remap (mu + alpha*(x-mean)) with a pure random constant
    per spatial region: y[mask] = mu.  No alpha term — the intensity ordering within
    each region is fully destroyed.

    This directly mimics what SynthSeg-modeB does (uniform GMM intensity per EM
    cluster) using our own K-means + Voronoi parcellation instead of sklearn GMM.
    Expected to achieve similar recall/coverage to SynthSeg-modeB but with our
    parcellation's spatial structure.
    """

    def __call__(self, input_images, hist_module, labels=None, **kwargs):
        images = input_images
        B, C_ch, D, H, W = images.shape
        device = images.device
        dtype  = images.dtype
        mask   = images > 0.01
        y      = images.clone()

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=False):
            images_f = images.float()
            for b in range(B):
                b_mask = mask[b, 0]
                if not b_mask.any():
                    continue
                img_b   = images_f[b, 0]
                fg_vals = img_b[b_mask]
                y_b     = y[b, 0]

                if torch.rand((1,), device=device).item() < self.SKIP_PARCELLATION_PROB:
                    mu = torch.rand((1,), device=device).item()
                    y_b[b_mask] = torch.full_like(img_b[b_mask], mu)
                    continue

                c_idx     = int(torch.rand((1,), device=device).item() * len(self.C_CHOICES))
                C         = self.C_CHOICES[min(c_idx, len(self.C_CHOICES) - 1)]
                N_fg      = fg_vals.shape[0]
                sub_fg    = fg_vals[torch.randperm(N_fg, device=device)[:min(N_fg, 10000)]]
                centroids = torch.linspace(sub_fg.min(), sub_fg.max(), C, device=device)
                for _ in range(10):
                    dists = torch.abs(sub_fg.unsqueeze(1) - centroids.unsqueeze(0))
                    lbls  = torch.argmin(dists, dim=1)
                    sums  = torch.zeros(C, device=device).scatter_add_(0, lbls, sub_fg)
                    cnts  = torch.zeros(C, device=device).scatter_add_(0, lbls, torch.ones_like(sub_fg))
                    new_c = torch.where(cnts > 0, sums / cnts, centroids)
                    if torch.allclose(centroids, new_c): break
                    centroids = new_c

                dists_all  = torch.abs(img_b.unsqueeze(-1) - centroids.view(1, 1, 1, C))
                class_lbls = torch.argmin(dists_all, dim=-1)
                coords     = torch.stack(torch.meshgrid(
                    torch.arange(D, device=device),
                    torch.arange(H, device=device),
                    torch.arange(W, device=device),
                    indexing="ij"), dim=-1).float()

                for c in range(C):
                    c_mask = b_mask & (class_lbls == c)
                    N_c    = c_mask.sum()
                    if N_c == 0:
                        continue
                    if torch.rand((1,), device=device).item() < self.SKIP_SUB_PARC_PROB:
                        S = 1
                    else:
                        s_idx = int(torch.rand((1,), device=device).item() * len(self.S_CHOICES))
                        S = min(self.S_CHOICES[min(s_idx, len(self.S_CHOICES) - 1)], N_c.item())
                    if S == 0:
                        continue
                    if S == 1:
                        mu = torch.rand((1,), device=device).item()
                        y_b[c_mask] = torch.full_like(img_b[c_mask], mu)
                        continue
                    c_coords    = coords[c_mask]
                    c_centroids = c_coords[torch.randperm(N_c, device=device)[:S]]
                    sub_lbls    = torch.argmin(torch.cdist(c_coords, c_centroids), dim=1)
                    sl_flat     = torch.full((D * H * W,), -1, dtype=torch.long, device=device)
                    sl_flat[c_mask.view(-1)] = sub_lbls
                    sub_lbls_3d = sl_flat.view(D, H, W)
                    for s in range(S):
                        s_mask = c_mask & (sub_lbls_3d == s)
                        if not s_mask.any():
                            continue
                        mu = torch.rand((1,), device=device).item()
                        y_b[s_mask] = torch.full_like(img_b[s_mask], mu)

        y = _normalize_guidance(y.float(), mask.float()).to(dtype)
        y = torch.where(mask, y, torch.zeros_like(y))
        return y, y, y


# ─── v26_8: signed alpha + global intensity inversion (50 %) ──────────────────

class V26_8GlobalInversionTargetGenerator(V26_6SignedAlphaTargetGenerator):
    """V26_8: V26_6 parcellation followed by per-sample global intensity inversion.

    After the signed-alpha parcellation and min-max normalisation, flip the
    entire brightness scale within the brain mask with probability 0.5:
        y_out = 1 - y_in   (within mask)

    Effect: creates two disjoint OOD archetype families (T1w-polarity and
    anti-T1w-polarity) that are completely separate in feature space, roughly
    doubling the effective number of OOD modes → higher Vendi.
    Coverage is preserved because the non-inverted 50 % still covers all
    real-cluster groups.
    """

    INVERSION_PROB: float = 0.5

    def __call__(self, input_images, hist_module, labels=None, **kwargs):
        y, _, _ = super().__call__(input_images, hist_module, labels=None, **kwargs)
        mask = input_images > 0.01
        B = y.shape[0]
        device = y.device

        for b in range(B):
            if torch.rand(1, device=device).item() < self.INVERSION_PROB:
                b_mask = mask[b, 0]
                y[b, 0] = torch.where(b_mask, 1.0 - y[b, 0], y[b, 0])

        return y, y, y


# ─── v26_9: signed alpha + random gamma tone curve ────────────────────────────

class V26_9GammaToneTargetGenerator(V26_6SignedAlphaTargetGenerator):
    """V26_9: V26_6 parcellation followed by per-sample log-uniform gamma correction.

    After the signed-alpha parcellation and min-max normalisation, apply:
        y_out = y_in ^ gamma    (within brain mask only)
    where gamma ~ LogUniform(GAMMA_LO, GAMMA_HI).

    Low gamma (< 1): spreads dark regions, compresses bright → "bright" images.
    High gamma (> 1): spreads bright, compresses dark → "dark" images.
    gamma = 1: identity (no change).

    y_in ∈ [0, 1] guarantees y_out ∈ [0, 1] for any gamma > 0, so no
    re-normalisation is needed.  The wide gamma range creates a much broader
    family of global-histogram archetypes → substantially higher Vendi.
    """

    GAMMA_LO: float = 0.25
    GAMMA_HI: float = 4.0

    def __call__(self, input_images, hist_module, labels=None, **kwargs):
        import math
        y, _, _ = super().__call__(input_images, hist_module, labels=None, **kwargs)
        mask = input_images > 0.01
        B = y.shape[0]
        device = y.device
        log_lo, log_hi = math.log(self.GAMMA_LO), math.log(self.GAMMA_HI)

        for b in range(B):
            gamma = math.exp(
                torch.rand(1, device=device).item() * (log_hi - log_lo) + log_lo
            )
            b_mask = mask[b, 0]
            y[b, 0] = torch.where(b_mask, y[b, 0].clamp(0.0, 1.0).pow(gamma), y[b, 0])

        return y, y, y


# ─── v26_10: signed alpha + additive fractal noise ────────────────────────────

class V26_10FractalNoiseTargetGenerator(V26_6SignedAlphaTargetGenerator):
    """V26_10: V26_6 parcellation plus per-sample additive fractal noise.

    After the signed-alpha parcellation and min-max normalisation, add
    spatially-correlated fractal noise scaled by sigma ~ U(SIGMA_LO, SIGMA_HI):
        y_out = clamp(y_in + sigma * noise, 0, 1)   (within brain mask)
    where noise is generated by generate_fractal_noise_3d (already available).

    Effect:
    - Each image gets a unique spatial texture fingerprint → higher Vendi.
    - The texture variation may improve HOG-space coverage (the one space
      where all intensity-remapping methods currently fail).
    - Small sigma keeps the parcellation structure intact; large sigma starts
      to approach a "noise-only" image (controlled by SIGMA_HI).
    """

    SIGMA_LO: float = 0.05
    SIGMA_HI: float = 0.25

    def __call__(self, input_images, hist_module, labels=None, **kwargs):
        y, _, _ = super().__call__(input_images, hist_module, labels=None, **kwargs)
        mask = input_images > 0.01
        B = input_images.shape[0]
        device = y.device

        # generate_fractal_noise_3d returns (B,1,D,H,W) in [-0.5, 0.5]
        noise_batch = generate_fractal_noise_3d(
            reference=input_images.float(), scales=(2, 4, 8, 16)
        )

        for b in range(B):
            sigma = (
                torch.rand(1, device=device).item() * (self.SIGMA_HI - self.SIGMA_LO)
                + self.SIGMA_LO
            )
            b_mask = mask[b, 0]
            noise = noise_batch[b, 0]          # D×H×W in [-0.5, 0.5]
            y[b, 0] = torch.where(
                b_mask,
                (y[b, 0] + sigma * noise).clamp(0.0, 1.0),
                y[b, 0],
            )

        return y, y, y


# ─── v28 family — HOG-space improvement ──────────────────────────────────────

class V28_2ResolutionDiversityTargetGenerator(V26_6SignedAlphaTargetGenerator):
    """V28_2: V26_6 signed-alpha parcellation, no generator-level change.

    Resolution diversity is applied at synthesis time via --resolution-diversity flag
    in generate_synthetic_guidance.py, which saves 50% of subjects at 2–4 mm voxel
    size so the HOG3D extractor sees genuinely coarse-resolution volumes.
    This class is identical to V26_6 — the generator config exists only to give
    the version a distinct name/checkpoint lookup.
    """
    pass


class V28_3SusceptibilityTargetGenerator(V26_6SignedAlphaTargetGenerator):
    """V28_3: V26_6 + synthetic susceptibility signal dropout.

    After the signed-alpha parcellation, applies a spatially varying signal
    attenuation mask that mimics B0 susceptibility (χ) effects in GRE images:
      • Inferior brain (z = 0..0.30 fraction of crop height): strong dropout
        simulating orbitofrontal and inferior temporal signal voids.
      • Lateral edges (|x| > 0.35 fraction of crop width): moderate dropout
        simulating temporal-pole susceptibility.

    The dropout creates sharp gradient discontinuities at the boundary zones,
    which generate strong near-vertical HOG features in inferior-brain cells
    — exactly the signature (hog3d_c*23_o7 / hog3d_c*13_o7) that most
    discriminates GRE from other modalities.

    Parameters:
      SUSCEPT_PROB      Probability of applying susceptibility per image.
      MAX_DROPOUT       Maximum signal fraction removed (clipped to [0,1]).
      INFERIOR_FRAC     Fraction of crop height treated as "inferior zone".
    """

    SUSCEPT_PROB: float    = 0.7
    MAX_DROPOUT: float     = 0.80
    INFERIOR_FRAC: float   = 0.30

    def __call__(self, input_images, hist_module, labels=None, **kwargs):
        y, _, _ = super().__call__(input_images, hist_module, labels=None, **kwargs)
        mask   = input_images > 0.01
        B, _, D, H, W = input_images.shape
        device = y.device

        for b in range(B):
            if torch.rand(1, device=device).item() > self.SUSCEPT_PROB:
                continue

            b_mask = mask[b, 0]
            max_drop = torch.rand(1, device=device).item() * self.MAX_DROPOUT

            # z-axis: 0 = inferior, D-1 = superior (in RAS crop)
            z_frac = torch.arange(D, device=device).float() / max(D - 1, 1)
            # Inferior dropout: sigmoid centred at INFERIOR_FRAC
            inf_drop = torch.sigmoid(-30.0 * (z_frac - self.INFERIOR_FRAC))  # (D,)

            # x-axis lateral dropout (temporal poles)
            x_frac = (torch.arange(W, device=device).float() / max(W - 1, 1) - 0.5).abs()
            lat_drop = torch.sigmoid(25.0 * (x_frac - 0.38))  # (W,)

            # Combined dropout map: dominant inferior, secondary lateral
            dropout_3d = (
                inf_drop.view(D, 1, 1) * 0.75 +
                lat_drop.view(1, 1, W) * 0.25
            ).clamp(0.0, 1.0) * max_drop              # (D, 1, W) broadcast → (D, H, W)

            y_b = y[b, 0]
            y[b, 0] = torch.where(b_mask, y_b * (1.0 - dropout_3d), y_b)

        return y, y, y


class V28_4CombinedTargetGenerator(V28_3SusceptibilityTargetGenerator):
    """V28_4: V28_3 susceptibility + resolution diversity at save time.

    Target generator identical to V28_3 (susceptibility dropout on top of V26_6).
    Resolution diversity (saving at 2–4 mm) is applied at synthesis time via
    the --resolution-diversity flag in generate_synthetic_guidance.py.

    This is the full v28 combination: HOG improvement via both
    (1) susceptibility gradient patterns and (2) genuine coarse-resolution saves.
    """
    pass


class V28_1RicianNoiseTargetGenerator(V26_6SignedAlphaTargetGenerator):
    """V28_1: V26_6 signed-alpha parcellation + per-sample Rician noise.

    After the signed-alpha parcellation and min-max normalisation, injects
    Rician (magnitude-MRI) noise:
        y_noisy = sqrt( (y + eps_r)^2 + eps_i^2 )
    where eps_r, eps_i ~ N(0, sigma), sigma ~ U(0, SIGMA_MAX).

    Combined with the aggressive resolution downsampling applied at the
    guidance-synthesis stage (zoom ∈ U(0.20, 1.0) set in
    generate_synthetic_guidance.py), this targets the HOG3D gap in
    bold/DWI/EPI modalities by:
      1. Zoom 0.20-0.40 → effective 2.5-5 mm resolution → coarser gradient
         patterns matching native bold/EPI at 3-4 mm.
      2. Rician noise → realistic noise texture that changes gradient
         magnitude distributions, improving match with real acquisitions.

    Regional_hist_64 impact is expected to be small: blurring mixes
    tissue-boundary intensities (partial volume), which actually makes the
    regional histograms more similar to bold/DWI (good), while noise at
    sigma ≤ 0.12 shifts histogram bins only slightly.
    """

    SIGMA_MAX: float = 0.12

    def __call__(self, input_images, hist_module, labels=None, **kwargs):
        y, _, _ = super().__call__(input_images, hist_module, labels=None, **kwargs)
        mask   = input_images > 0.01
        B      = y.shape[0]
        device = y.device

        for b in range(B):
            sigma = torch.rand(1, device=device).item() * self.SIGMA_MAX
            if sigma < 1e-4:
                continue
            y_b   = y[b, 0]
            b_mask = mask[b, 0]
            # Rician: magnitude of complex Gaussian with signal y and noise sigma
            eps_r = torch.randn_like(y_b) * sigma
            eps_i = torch.randn_like(y_b) * sigma
            y_noisy = torch.sqrt((y_b + eps_r).pow(2) + eps_i.pow(2))
            # Re-normalise brain to [0,1] after noise (noise can push values > 1)
            brain_vals = y_noisy[b_mask]
            lo = brain_vals.min()
            hi = brain_vals.max()
            if hi > lo:
                y_noisy = (y_noisy - lo) / (hi - lo + 1e-6)
            y[b, 0] = torch.where(b_mask, y_noisy.clamp(0.0, 1.0), torch.zeros_like(y_noisy))

        return y, y, y


# ─── Shared helper: 1-D K-means ────────────────────────────────────────────────

def _kmeans_1d(fg_vals: torch.Tensor, C: int) -> torch.Tensor:
    """Return centroids (shape C) for 1-D K-means on fg_vals (max 10 k subsampled)."""
    device = fg_vals.device
    n = fg_vals.shape[0]
    sub = fg_vals[torch.randperm(n, device=device)[:min(n, 10000)]]
    centroids = torch.linspace(sub.min(), sub.max(), C, device=device)
    for _ in range(10):
        dists = torch.abs(sub.unsqueeze(1) - centroids.unsqueeze(0))
        lbls  = torch.argmin(dists, dim=1)
        sums  = torch.zeros(C, device=device).scatter_add_(0, lbls, sub)
        cnts  = torch.zeros(C, device=device).scatter_add_(0, lbls, torch.ones_like(sub))
        new_c = torch.where(cnts > 0, sums / cnts, centroids)
        if torch.allclose(centroids, new_c): break
        centroids = new_c
    return centroids


# ─── v26_11: large-C signed-alpha parcellation, no Voronoi ────────────────────

class V26_11LargeKTargetGenerator(V26_4EMParcellationTargetGenerator):
    """V26_11: many K-means classes (8–16) + signed alpha, no Voronoi sub-parc.

    Replaces the small-C [2–6] K-means of V26_4 with a larger cluster count
    [8, 10, 12, 16], giving each image up to 16 independently remapped intensity
    zones.  Spatial Voronoi is disabled — each zone is treated as one region.

    More independent contrast knobs → samples land in more orthogonal positions
    in PCA space → higher effective number of OOD modes → higher Vendi.
    Signed alpha maintains within-region diversity (no collapse to flat regions).
    """

    C_CHOICES_LARGE = [8, 10, 12, 16]
    SKIP_PARCELLATION_PROB = 0.05

    def __call__(self, input_images, hist_module, labels=None, **kwargs):
        images = input_images
        B, C_ch, D, H, W = images.shape
        device = images.device
        dtype  = images.dtype
        mask   = images > 0.01
        y      = images.clone()

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=False):
            images_f = images.float()
            for b in range(B):
                b_mask = mask[b, 0]
                if not b_mask.any():
                    continue
                img_b   = images_f[b, 0]
                fg_vals = img_b[b_mask]
                y_b     = y[b, 0]

                def _signed_alpha() -> float:
                    sign = 1.0 if torch.rand(1, device=device).item() > 0.5 else -1.0
                    return sign * (torch.rand(1, device=device).item() * 1.5 + 0.5)

                if torch.rand(1, device=device).item() < self.SKIP_PARCELLATION_PROB:
                    mu    = torch.rand(1, device=device).item()
                    alpha = _signed_alpha()
                    y_b[b_mask] = (mu + alpha * (img_b[b_mask] - fg_vals.mean())).to(dtype)
                    continue

                c_idx = int(torch.rand(1, device=device).item() * len(self.C_CHOICES_LARGE))
                C     = self.C_CHOICES_LARGE[min(c_idx, len(self.C_CHOICES_LARGE) - 1)]
                centroids  = _kmeans_1d(fg_vals, C)
                dists_all  = torch.abs(img_b.unsqueeze(-1) - centroids.view(1, 1, 1, C))
                class_lbls = torch.argmin(dists_all, dim=-1)

                for c in range(C):
                    c_mask = b_mask & (class_lbls == c)
                    if not c_mask.any():
                        continue
                    mu    = torch.rand(1, device=device).item()
                    alpha = _signed_alpha()
                    y_b[c_mask] = (mu + alpha * (img_b[c_mask] - img_b[c_mask].mean())).to(dtype)

        y = _normalize_guidance(y.float(), mask.float()).to(dtype)
        y = torch.where(mask, y, torch.zeros_like(y))
        return y, y, y


# ─── v26_12: standard C, stratified mu, signed alpha per sub-region ───────────

class V26_12StratifiedMuTargetGenerator(V26_4EMParcellationTargetGenerator):
    """V26_12: V26_6-like parcellation with stratified class-level mu sampling.

    Instead of drawing mu ~ U(0,1) independently per region, we:
      1. Divide [0, 1] into C equal-width bins (one per K-means class).
      2. Draw one mu uniformly from each bin.
      3. Randomly SHUFFLE these C mus among the C classes.

    This guarantees that every image has one class targeting each brightness
    "zone" — the C targets always span the full [0,1] range.  Sub-regions
    within each class anchor around the class-level mu; signed alpha per
    sub-region provides within-class spread.

    Effect: OOD samples are forced to extremes in feature space (at least one
    very dark and one very bright class every time) → more distinct OOD
    positions → higher Vendi, while signed alpha keeps real-cluster coverage.
    """

    SKIP_PARCELLATION_PROB = 0.05
    SKIP_SUB_PARC_PROB     = 0.40

    def __call__(self, input_images, hist_module, labels=None, **kwargs):
        images = input_images
        B, C_ch, D, H, W = images.shape
        device = images.device
        dtype  = images.dtype
        mask   = images > 0.01
        y      = images.clone()

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=False):
            images_f = images.float()
            for b in range(B):
                b_mask = mask[b, 0]
                if not b_mask.any():
                    continue
                img_b   = images_f[b, 0]
                fg_vals = img_b[b_mask]
                y_b     = y[b, 0]

                def _signed_alpha() -> float:
                    sign = 1.0 if torch.rand(1, device=device).item() > 0.5 else -1.0
                    return sign * (torch.rand(1, device=device).item() * 1.5 + 0.5)

                if torch.rand(1, device=device).item() < self.SKIP_PARCELLATION_PROB:
                    mu    = torch.rand(1, device=device).item()
                    alpha = _signed_alpha()
                    y_b[b_mask] = (mu + alpha * (img_b[b_mask] - fg_vals.mean())).to(dtype)
                    continue

                c_idx = int(torch.rand(1, device=device).item() * len(self.C_CHOICES))
                C     = self.C_CHOICES[min(c_idx, len(self.C_CHOICES) - 1)]
                centroids  = _kmeans_1d(fg_vals, C)
                dists_all  = torch.abs(img_b.unsqueeze(-1) - centroids.view(1, 1, 1, C))
                class_lbls = torch.argmin(dists_all, dim=-1)
                coords     = torch.stack(torch.meshgrid(
                    torch.arange(D, device=device),
                    torch.arange(H, device=device),
                    torch.arange(W, device=device),
                    indexing="ij"), dim=-1).float()

                # Stratified mu: one sample per [k/C, (k+1)/C] bin, shuffled
                bin_w  = 1.0 / C
                mu_per_class = torch.zeros(C, device=device)
                perm   = torch.randperm(C, device=device)
                for k in range(C):
                    lo = k * bin_w
                    mu_per_class[perm[k]] = lo + torch.rand(1, device=device).item() * bin_w

                for c in range(C):
                    c_mask   = b_mask & (class_lbls == c)
                    N_c      = c_mask.sum()
                    if N_c == 0:
                        continue
                    class_mu = mu_per_class[c].item()
                    if torch.rand(1, device=device).item() < self.SKIP_SUB_PARC_PROB:
                        S = 1
                    else:
                        s_idx = int(torch.rand(1, device=device).item() * len(self.S_CHOICES))
                        S = min(self.S_CHOICES[min(s_idx, len(self.S_CHOICES) - 1)], N_c.item())
                    if S == 0:
                        continue
                    if S == 1:
                        alpha = _signed_alpha()
                        y_b[c_mask] = (class_mu + alpha * (img_b[c_mask] - img_b[c_mask].mean())).to(dtype)
                        continue
                    c_coords    = coords[c_mask]
                    c_centroids = c_coords[torch.randperm(N_c, device=device)[:S]]
                    sub_lbls    = torch.argmin(torch.cdist(c_coords, c_centroids), dim=1)
                    sl_flat     = torch.full((D * H * W,), -1, dtype=torch.long, device=device)
                    sl_flat[c_mask.view(-1)] = sub_lbls
                    sub_lbls_3d = sl_flat.view(D, H, W)
                    for s in range(S):
                        s_mask = c_mask & (sub_lbls_3d == s)
                        if not s_mask.any():
                            continue
                        alpha = _signed_alpha()
                        y_b[s_mask] = (class_mu + alpha * (img_b[s_mask] - img_b[s_mask].mean())).to(dtype)

        y = _normalize_guidance(y.float(), mask.float()).to(dtype)
        y = torch.where(mask, y, torch.zeros_like(y))
        return y, y, y


# ─── v26_13: large C + stratified mu, no Voronoi ──────────────────────────────

class V26_13LargeKStratifiedTargetGenerator(V26_4EMParcellationTargetGenerator):
    """V26_13: large-C (8–16) K-means + stratified mu + signed alpha, no sub-parc.

    Combines V26_11 (large class count) and V26_12 (stratified mu sampling):
      - C ∈ {8, 10, 12, 16} classes → many independent contrast zones.
      - Stratified mu: each class targets a distinct [k/C, (k+1)/C] brightness band.
      - Signed alpha: within-class inversion for additional diversity.
      - No Voronoi sub-parcellation (kept simple given large C).

    Expected: highest diversity of all v26_* variants — large C × stratified mu
    creates an enormous number of distinct macro-contrast archetypes.
    """

    C_CHOICES_LARGE      = [8, 10, 12, 16]
    SKIP_PARCELLATION_PROB = 0.05

    def __call__(self, input_images, hist_module, labels=None, **kwargs):
        images = input_images
        B, C_ch, D, H, W = images.shape
        device = images.device
        dtype  = images.dtype
        mask   = images > 0.01
        y      = images.clone()

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=False):
            images_f = images.float()
            for b in range(B):
                b_mask = mask[b, 0]
                if not b_mask.any():
                    continue
                img_b   = images_f[b, 0]
                fg_vals = img_b[b_mask]
                y_b     = y[b, 0]

                def _signed_alpha() -> float:
                    sign = 1.0 if torch.rand(1, device=device).item() > 0.5 else -1.0
                    return sign * (torch.rand(1, device=device).item() * 1.5 + 0.5)

                if torch.rand(1, device=device).item() < self.SKIP_PARCELLATION_PROB:
                    mu    = torch.rand(1, device=device).item()
                    alpha = _signed_alpha()
                    y_b[b_mask] = (mu + alpha * (img_b[b_mask] - fg_vals.mean())).to(dtype)
                    continue

                c_idx = int(torch.rand(1, device=device).item() * len(self.C_CHOICES_LARGE))
                C     = self.C_CHOICES_LARGE[min(c_idx, len(self.C_CHOICES_LARGE) - 1)]
                centroids  = _kmeans_1d(fg_vals, C)
                dists_all  = torch.abs(img_b.unsqueeze(-1) - centroids.view(1, 1, 1, C))
                class_lbls = torch.argmin(dists_all, dim=-1)

                # Stratified mu: bin [k/C, (k+1)/C] → class perm[k]
                bin_w  = 1.0 / C
                mu_per_class = torch.zeros(C, device=device)
                perm   = torch.randperm(C, device=device)
                for k in range(C):
                    lo = k * bin_w
                    mu_per_class[perm[k]] = lo + torch.rand(1, device=device).item() * bin_w

                for c in range(C):
                    c_mask = b_mask & (class_lbls == c)
                    if not c_mask.any():
                        continue
                    mu    = mu_per_class[c].item()
                    alpha = _signed_alpha()
                    y_b[c_mask] = (mu + alpha * (img_b[c_mask] - img_b[c_mask].mean())).to(dtype)

        y = _normalize_guidance(y.float(), mask.float()).to(dtype)
        y = torch.where(mask, y, torch.zeros_like(y))
        return y, y, y


# ─── v26_15: compound double remap ───────────────────────────────────────────

class V26_15DoubleRemapTargetGenerator(V26_6SignedAlphaTargetGenerator):
    """V26_15: Apply the signed-alpha V26_6 parcellation twice in sequence.

    Round 1: K-means on the T1w input → y1   (V26_6 remap, normalized to [0,1])
    Round 2: K-means on y1 → y2              (another V26_6 remap on y1, normalized)

    Because the second K-means sees y1's distribution (already remapped), it
    parcellates different intensity boundaries than round 1 — the two passes
    compose non-linearly.  The compound transformation can:
      • Double-invert (T1w-like → T2w-like → T1w-like again)
      • Cross-transform (T1w-like → T2w-like → FLAIR-like)
      • Amplify contrasts far beyond the single-remap dynamic range

    Expected effect: wider coverage of the intensity manifold, higher Vendi
    (more compound archetypes), at ≈ 2× the compute cost of V26_6.
    """

    def __call__(self, input_images, hist_module, labels=None, **kwargs):
        y1, _, _ = super().__call__(input_images, hist_module, labels=None, **kwargs)
        y2, _, _ = super().__call__(y1,            hist_module, labels=None, **kwargs)
        return y2, y2, y2


# ─── v26_14: mixed affine+flat per class ─────────────────────────────────────

class V26_14MixedFlatTargetGenerator(V26_4EMParcellationTargetGenerator):
    """V26_14: per-class random choice of affine remap (V26_6) vs flat constant.

    For each K-means class, independently draw the remap mode:
      - With prob FLAT_PROB: assign a flat random constant (y[mask] = mu).
        This makes the class's histogram a delta spike → pushes the sample
        toward a corner of PCA space → higher Vendi.
      - With prob 1-FLAT_PROB: apply the usual signed-alpha affine remap
        (mu + alpha*(x - mean)).  Preserves smooth within-class histograms
        → maintains cluster coverage for real modality × scanner groups.

    The per-class independence means each image has a different mix of flat
    and affine regions, creating many more distinct OOD archetypes while
    keeping the average coverage high (affine regions still match real
    cluster structures).
    """

    FLAT_PROB:             float = 0.4
    SKIP_PARCELLATION_PROB: float = 0.05
    SKIP_SUB_PARC_PROB:     float = 0.40

    def __call__(self, input_images, hist_module, labels=None, **kwargs):
        images = input_images
        B, C_ch, D, H, W = images.shape
        device = images.device
        dtype  = images.dtype
        mask   = images > 0.01
        y      = images.clone()

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=False):
            images_f = images.float()
            for b in range(B):
                b_mask = mask[b, 0]
                if not b_mask.any():
                    continue
                img_b   = images_f[b, 0]
                fg_vals = img_b[b_mask]
                y_b     = y[b, 0]

                def _signed_alpha() -> float:
                    sign = 1.0 if torch.rand(1, device=device).item() > 0.5 else -1.0
                    return sign * (torch.rand(1, device=device).item() * 1.5 + 0.5)

                if torch.rand(1, device=device).item() < self.SKIP_PARCELLATION_PROB:
                    mu    = torch.rand(1, device=device).item()
                    alpha = _signed_alpha()
                    y_b[b_mask] = (mu + alpha * (img_b[b_mask] - fg_vals.mean())).to(dtype)
                    continue

                c_idx = int(torch.rand(1, device=device).item() * len(self.C_CHOICES))
                C     = self.C_CHOICES[min(c_idx, len(self.C_CHOICES) - 1)]
                centroids  = _kmeans_1d(fg_vals, C)
                dists_all  = torch.abs(img_b.unsqueeze(-1) - centroids.view(1, 1, 1, C))
                class_lbls = torch.argmin(dists_all, dim=-1)
                coords     = torch.stack(torch.meshgrid(
                    torch.arange(D, device=device),
                    torch.arange(H, device=device),
                    torch.arange(W, device=device),
                    indexing="ij"), dim=-1).float()

                for c in range(C):
                    c_mask = b_mask & (class_lbls == c)
                    N_c    = c_mask.sum()
                    if N_c == 0:
                        continue
                    use_flat = torch.rand(1, device=device).item() < self.FLAT_PROB
                    mu = torch.rand(1, device=device).item()

                    if use_flat:
                        # Flat constant for this class — all sub-regions get same value
                        y_b[c_mask] = torch.full_like(img_b[c_mask], mu)
                        continue

                    # Signed-alpha affine remap (with optional sub-parcellation)
                    if torch.rand(1, device=device).item() < self.SKIP_SUB_PARC_PROB:
                        S = 1
                    else:
                        s_idx = int(torch.rand(1, device=device).item() * len(self.S_CHOICES))
                        S = min(self.S_CHOICES[min(s_idx, len(self.S_CHOICES) - 1)], N_c.item())
                    if S == 0:
                        continue
                    if S == 1:
                        alpha = _signed_alpha()
                        y_b[c_mask] = (mu + alpha * (img_b[c_mask] - img_b[c_mask].mean())).to(dtype)
                        continue
                    c_coords    = coords[c_mask]
                    c_centroids = c_coords[torch.randperm(N_c, device=device)[:S]]
                    sub_lbls    = torch.argmin(torch.cdist(c_coords, c_centroids), dim=1)
                    sl_flat     = torch.full((D * H * W,), -1, dtype=torch.long, device=device)
                    sl_flat[c_mask.view(-1)] = sub_lbls
                    sub_lbls_3d = sl_flat.view(D, H, W)
                    for s in range(S):
                        s_mask = c_mask & (sub_lbls_3d == s)
                        if not s_mask.any():
                            continue
                        alpha = _signed_alpha()
                        y_b[s_mask] = (mu + alpha * (img_b[s_mask] - img_b[s_mask].mean())).to(dtype)

        y = _normalize_guidance(y.float(), mask.float()).to(dtype)
        y = torch.where(mask, y, torch.zeros_like(y))
        return y, y, y


# ─── Mode A: label-conditioned per-region quantile chunking ───────────────────

class V27ALabConditionedTargetGenerator(BaseTargetGenerator):
    """
    Mode A: Label-conditioned histogram remapping.

    For each anatomical label region (from an external segmentation):
      1. Draw K from K_CHOICES.
      2. Compute K intensity quantile chunks WITHIN that label's voxels.
      3. Apply an independent (mu, alpha) affine remap per chunk.

    Each anatomical region gets fully independent contrast — the boundaries
    between regions can have arbitrary intensity discontinuities.

    Falls back to a single global affine remap if no label map is provided.
    Background label (0) is always skipped.
    """

    K_CHOICES: list[int] = [2, 3, 4, 6, 8]
    MIN_VOXELS: int = 100

    def __call__(
        self,
        input_images: torch.Tensor,
        hist_module: nn.Module,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = input_images
        B, C_ch, D, H, W = images.shape
        device = images.device
        dtype = images.dtype
        mask = images > 0.01
        y = images.clone()

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=False):
            images_f = images.float()

            if labels is not None:
                lbl = labels
                if lbl.dim() == 4:
                    lbl = lbl.unsqueeze(1)
                if lbl.shape[2:] != images.shape[2:]:
                    lbl = F.interpolate(lbl.float(), size=images.shape[2:], mode="nearest")
                lbl = lbl.long()

            for b in range(B):
                b_mask = mask[b, 0]
                img_b = images_f[b, 0]
                y_b = img_b.clone()

                if labels is None:
                    # Fallback: single global affine remap
                    fg_vals = img_b[b_mask]
                    mu = torch.rand(1, device=device).item()
                    alpha = torch.rand(1, device=device).item() * 1.5 + 0.5
                    y_b[b_mask] = mu + alpha * (fg_vals - fg_vals.mean())
                else:
                    lbl_map = lbl[b, 0]
                    label_ids = lbl_map[b_mask].unique()

                    for lbl_id in label_ids:
                        if lbl_id.item() == 0:
                            continue
                        lbl_mask = b_mask & (lbl_map == lbl_id)
                        N = lbl_mask.sum().item()
                        if N < self.MIN_VOXELS:
                            continue

                        k_idx = int(torch.rand(1, device=device).item() * len(self.K_CHOICES))
                        K = self.K_CHOICES[min(k_idx, len(self.K_CHOICES) - 1)]

                        lbl_vals = img_b[lbl_mask]
                        probs = torch.linspace(0.0, 1.0, K + 1, device=device)
                        q_edges = torch.quantile(lbl_vals, probs)
                        q_edges[0]  = lbl_vals.min() - 1e-6
                        q_edges[-1] = lbl_vals.max() + 1e-6

                        chunk_ids = torch.full((D, H, W), -1, dtype=torch.long, device=device)
                        c_idx = (torch.bucketize(lbl_vals, q_edges) - 1).clamp(0, K - 1)
                        chunk_ids[lbl_mask] = c_idx

                        for k in range(K):
                            k_mask = lbl_mask & (chunk_ids == k)
                            if not k_mask.any():
                                continue
                            chunk_vals = img_b[k_mask]
                            mu = torch.rand(1, device=device).item()
                            alpha = torch.rand(1, device=device).item() * 1.5 + 0.5
                            y_b[k_mask] = mu + alpha * (chunk_vals - chunk_vals.mean())

                y[b, 0] = y_b.to(dtype)

        y = _normalize_guidance(y.float(), mask.float()).to(dtype)
        y = torch.where(mask, y, torch.zeros_like(y))
        target_hist = hist_module(y)
        return target_hist, y, y


# ─── Mode A bis: global EM parcellation + per-selected-label refinement ───────

class V27ABisTargetGenerator(V26_4EMParcellationTargetGenerator):
    """
    Mode A bis: Global EM parcellation (Mode B / V26_4) followed by an
    additional per-selected-label intra-label affine remap.

    Step 1 — Run V26_4 EM parcellation on the whole brain (inherited).
    Step 2 — For each anatomical label (from external seg), with probability
              LABEL_SELECT_PROB, apply one additional affine remap on the
              already-parcellated intensities within that label.  This creates
              a two-level contrast: global structure from EM parcellation +
              local modulation locked to anatomical boundaries.

    Falls back to plain V26_4 behaviour if no label map is provided.
    Background label (0) is always skipped.
    """

    LABEL_SELECT_PROB: float = 0.5
    MIN_VOXELS: int = 100

    def __call__(
        self,
        input_images: torch.Tensor,
        hist_module: nn.Module,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Step 1: V26_4 global EM parcellation (returns normalised guidance)
        y_em, _, _ = super().__call__(input_images, hist_module, labels=None, **kwargs)

        if labels is None:
            return hist_module(y_em), y_em, y_em

        B, C_ch, D, H, W = input_images.shape
        mask = input_images > 0.01
        device = input_images.device
        dtype = input_images.dtype

        lbl = labels
        if lbl.dim() == 4:
            lbl = lbl.unsqueeze(1)
        if lbl.shape[2:] != input_images.shape[2:]:
            lbl = F.interpolate(lbl.float(), size=input_images.shape[2:], mode="nearest")
        lbl = lbl.long()

        y_f = y_em.float()
        for b in range(B):
            b_mask = mask[b, 0]
            y_b = y_f[b, 0]
            lbl_map = lbl[b, 0]
            label_ids = lbl_map[b_mask].unique()

            for lbl_id in label_ids:
                if lbl_id.item() == 0:
                    continue
                if torch.rand(1, device=device).item() > self.LABEL_SELECT_PROB:
                    continue
                lbl_mask = b_mask & (lbl_map == lbl_id)
                if lbl_mask.sum().item() < self.MIN_VOXELS:
                    continue
                lbl_vals = y_b[lbl_mask]
                mu = torch.rand(1, device=device).item()
                alpha = torch.rand(1, device=device).item() * 1.5 + 0.5
                y_b[lbl_mask] = mu + alpha * (lbl_vals - lbl_vals.mean())

            y_f[b, 0] = y_b

        y = _normalize_guidance(y_f, mask.float()).to(dtype)
        y = torch.where(mask, y, torch.zeros_like(y))
        return hist_module(y), y, y
