from __future__ import annotations

import time

import torch

from src.synthesis.noise_ops import generate_fractal_noise_3d


def test_generate_fractal_noise_3d_shape_range_and_finite() -> None:
    x = torch.rand((2, 1, 64, 64, 64), dtype=torch.float32)
    noise = generate_fractal_noise_3d(x)

    assert noise.shape == (2, 1, 64, 64, 64)
    assert torch.isfinite(noise).all()
    assert float(noise.min()) >= -0.51
    assert float(noise.max()) <= 0.51


def test_generate_fractal_noise_3d_is_differentiable_and_fast() -> None:
    x = torch.rand((1, 1, 48, 48, 48), dtype=torch.float32, requires_grad=True)

    start = time.perf_counter()
    noise = generate_fractal_noise_3d(x)
    elapsed = time.perf_counter() - start

    # Keep this generous to avoid flaky runtime failures on loaded machines.
    assert elapsed < 8.0

    loss = (noise * x).mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
