from __future__ import annotations

import time

import torch

from src.filters import AnatomicalUnsharpMask3D


def test_anatomical_unsharp_mask_3d_shape_clamp_and_grad() -> None:
    module = AnatomicalUnsharpMask3D(alpha=2.0, sigma=1.0)
    x = torch.rand((2, 1, 32, 32, 32), dtype=torch.float32, requires_grad=True)

    y = module(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert float(y.detach().min()) >= 0.0
    assert float(y.detach().max()) <= 1.0

    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_anatomical_unsharp_mask_3d_runtime_sanity() -> None:
    module = AnatomicalUnsharpMask3D(alpha=2.0, sigma=1.0)
    x = torch.rand((1, 1, 48, 48, 48), dtype=torch.float32)

    start = time.perf_counter()
    y = module(x)
    elapsed = time.perf_counter() - start

    assert y.shape == x.shape
    # Generous threshold to avoid flakes under load.
    assert elapsed < 4.0
