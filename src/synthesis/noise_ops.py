from __future__ import annotations

import torch
import torch.nn.functional as F


def generate_fractal_noise_3d(
    reference: torch.Tensor,
    scales: tuple[int, ...] = (2, 4, 8, 16),
    eps: float = 1e-6,
    noise_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Generate smooth multi-scale 3D fractal noise normalized to [-0.5, 0.5].

    Args:
        reference: 5D tensor shaped as (B, C, D, H, W). Only B/D/H/W are used.
        scales: Downsampling factors used to build multi-scale fields.
        eps: Numerical stability term for normalization.
    """
    if reference.ndim != 5:
        raise ValueError(f"Expected 5D tensor (B, C, D, H, W), got shape {tuple(reference.shape)}")

    b, _, d, h, w = reference.shape
    device = reference.device
    dtype = reference.dtype
    working_dtype = noise_dtype if noise_dtype is not None else dtype

    noise = torch.zeros((b, 1, d, h, w), device=device, dtype=working_dtype)
    weight_sum = 0.0

    for scale in scales:
        s = int(scale)
        if s <= 0:
            raise ValueError("All scales must be positive integers.")

        sd = max(1, d // s)
        sh = max(1, h // s)
        sw = max(1, w // s)

        low_res = torch.randn((b, 1, sd, sh, sw), device=device, dtype=working_dtype)
        upsampled = F.interpolate(
            low_res,
            size=(d, h, w),
            mode="trilinear",
            align_corners=False,
        )

        weight = 1.0 / float(s)
        noise.add_(upsampled, alpha=weight)
        weight_sum += weight

    noise = noise / max(weight_sum, eps)

    # Per-sample normalization to [0, 1], then shift to [-0.5, 0.5].
    n_min = noise.amin(dim=(-3, -2, -1), keepdim=True)
    n_max = noise.amax(dim=(-3, -2, -1), keepdim=True)
    normalized = (noise - n_min) / (n_max - n_min + eps)
    out = normalized - 0.5
    return out.to(dtype=dtype)
