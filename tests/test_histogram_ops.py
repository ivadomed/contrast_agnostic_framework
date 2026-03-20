from __future__ import annotations

import torch

from src.histogram_ops import DifferentiableHistogram3D, create_range_translation_guidance_map


def test_differentiable_histogram3d_shape_and_mass() -> None:
    hist = DifferentiableHistogram3D(num_bins=8, value_range=(0.0, 1.0))
    x = torch.tensor(
        [[[[[0.0, 0.25, 0.5, 0.75, 1.0]]]]],
        dtype=torch.float32,
    )

    out = hist(x)

    assert out.shape == (1, 1, 8)
    assert torch.isclose(out.sum(), torch.tensor(5.0), atol=1e-4)


def test_differentiable_histogram3d_supports_mask() -> None:
    hist = DifferentiableHistogram3D(num_bins=8, value_range=(0.0, 1.0))
    x = torch.tensor(
        [[[[[0.1, 0.2, 0.3, 0.4]]]]],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [[[[[1.0, 0.0, 1.0, 0.0]]]]],
        dtype=torch.float32,
    )

    out = hist(x, mask=mask)

    assert out.shape == (1, 1, 8)
    assert torch.isclose(out.sum(), torch.tensor(2.0), atol=1e-4)


def test_differentiable_histogram3d_is_differentiable() -> None:
    hist = DifferentiableHistogram3D(num_bins=16, value_range=(0.0, 1.0))
    x = torch.linspace(0.0, 1.0, 16, dtype=torch.float32).reshape(1, 1, 2, 2, 4)
    x.requires_grad_(True)

    out = hist(x)
    loss = out.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_create_range_translation_guidance_map_identity_perm_stability() -> None:
    x = torch.tensor(
        [
            [
                [
                    [[0.10, 0.20], [0.30, 0.40]],
                    [[0.50, 0.60], [0.70, 0.80]],
                ]
            ]
        ],
        dtype=torch.float32,
    )
    perms = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    guidance = create_range_translation_guidance_map(
        input_image=x,
        perms=perms,
        num_chunks=4,
        dark_threshold=0.05,
    )

    assert guidance.shape == x.shape
    assert torch.allclose(guidance, x, atol=1e-3)


def test_create_range_translation_guidance_map_keeps_valid_range() -> None:
    x = torch.tensor(
        [
            [
                [
                    [[0.02, 0.15], [0.25, 0.35]],
                    [[0.45, 0.55], [0.75, 0.95]],
                ]
            ]
        ],
        dtype=torch.float32,
    )
    perms = torch.tensor([[3, 2, 1, 0]], dtype=torch.long)

    guidance = create_range_translation_guidance_map(
        input_image=x,
        perms=perms,
        num_chunks=4,
        dark_threshold=0.05,
    )

    assert torch.all(guidance >= 0.0)
    assert torch.all(guidance <= 1.0)
