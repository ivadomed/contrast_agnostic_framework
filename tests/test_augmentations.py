from __future__ import annotations

import torch

from src.intensity_ops import (
    RandomAnisotropicDegradation3D,
    RandomBezierIntensityWarp,
    RandomSoftQuantileShuffling,
)


def test_random_bezier_intensity_warp_shape_bounds_and_grad() -> None:
    torch.manual_seed(0)
    module = RandomBezierIntensityWarp(p=1.0)
    x = torch.rand((2, 1, 24, 24, 24), dtype=torch.float32, requires_grad=True)

    y = module(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert float(y.detach().min()) >= 0.0
    assert float(y.detach().max()) <= 1.0

    loss = y.mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_random_anisotropic_degradation_3d_shape_bounds_and_grad() -> None:
    torch.manual_seed(0)
    module = RandomAnisotropicDegradation3D(p=1.0, min_factor=4, max_factor=4)
    x = torch.rand((2, 1, 32, 24, 24), dtype=torch.float32, requires_grad=True)

    y = module(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert float(y.detach().min()) >= 0.0
    assert float(y.detach().max()) <= 1.0

    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_random_soft_quantile_shuffling_shape_bounds_and_background() -> None:
    torch.manual_seed(0)
    module = RandomSoftQuantileShuffling(
        p=1.0,
        num_centroids=5,
        sample_size=5000,
        temperature=0.05,
        noise_std=0.02,
    )
    x = torch.zeros((2, 1, 24, 24, 24), dtype=torch.float32, requires_grad=True)
    x.data[:, :, 6:18, 6:18, 6:18] = torch.rand((2, 1, 12, 12, 12), dtype=torch.float32)

    y = module(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert float(y.detach().min()) >= 0.0
    assert float(y.detach().max()) <= 1.0

    background_mask = x == 0.0
    assert torch.all(y[background_mask] == 0.0)

    loss = y.mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
