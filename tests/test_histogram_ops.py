from __future__ import annotations

import time

import pytest
import torch

from src.synthesis.histogram_ops import DifferentiableHistogram3D, create_range_translation_guidance_map, generate_unified_targets
from src.synthesis.intensity_ops import RandomGMMHistogramMatching


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


def test_generate_unified_targets_v8_grid_chunking_shape_nan_and_compile() -> None:
    hist = DifferentiableHistogram3D(num_bins=16, value_range=(0.0, 1.0))
    x = torch.rand((2, 1, 16, 16, 16), dtype=torch.float32)

    target_hist, perms, guidance_map = generate_unified_targets(
        input_images=x,
        num_bins=16,
        num_chunks=8,
        dark_threshold=0.05,
        hist_module=hist,
        gen_version="v8",
        grid_size=(4, 4, 4),
    )

    assert target_hist.shape == (2, 1, 16)
    assert perms.shape == (2, 8)
    assert guidance_map.shape == x.shape
    assert torch.isfinite(guidance_map).all()

    if hasattr(torch, "compile"):
        def _compiled_forward(inp: torch.Tensor) -> torch.Tensor:
            _, _, g = generate_unified_targets(
                input_images=inp,
                num_bins=16,
                num_chunks=8,
                dark_threshold=0.05,
                hist_module=hist,
                gen_version="v8",
                grid_size=(4, 4, 4),
            )
            return g

        try:
            compiled_forward = torch.compile(_compiled_forward, mode="reduce-overhead")
            compiled_guidance = compiled_forward(x)
        except Exception as exc:
            pytest.skip(f"torch.compile not available in this environment: {exc}")

        assert compiled_guidance.shape == x.shape
        assert torch.isfinite(compiled_guidance).all()


def test_random_gmm_histogram_matching_bounds_and_speed() -> None:
    op = RandomGMMHistogramMatching(
        p=1.0,
        num_quantiles=100,
        sample_size=100000,
        num_bins=100,
        min_peaks=3,
        max_peaks=6,
    )
    x = torch.rand((2, 1, 64, 64, 64), dtype=torch.float32)

    start = time.perf_counter()
    out = op(x)
    elapsed = time.perf_counter() - start

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)
    # Keep a loose ceiling to avoid false negatives on slower CI machines.
    assert elapsed < 5.0


def test_random_gmm_histogram_matching_preserves_black_background() -> None:
    """Verify that black background pixels remain at 0 after histogram matching."""
    op = RandomGMMHistogramMatching(
        p=1.0,
        num_quantiles=100,
        sample_size=100000,
        num_bins=100,
        min_peaks=3,
        max_peaks=6,
    )
    
    # Create volume with black background (0) and white tissue (0.5-1.0)
    x = torch.zeros((2, 1, 32, 32, 32), dtype=torch.float32)
    x[:, :, 8:24, 8:24, 8:24] = torch.rand((2, 1, 16, 16, 16)) * 0.5 + 0.5  # tissue region [0.5, 1.0]
    
    out = op(x)
    
    # Check that all background pixels (originally 0) remain at 0
    background_mask = x == 0.0
    assert torch.all(out[background_mask] == 0.0), \
        f"Background pixels should remain at 0, but found min={out[background_mask].min()}, max={out[background_mask].max()}"
    
    # Check that tissue pixels are modified (not equal to input)
    tissue_mask = x > 0.0
    assert not torch.allclose(out[tissue_mask], x[tissue_mask]), \
        "Tissue pixels should be modified by histogram matching"


def test_generate_unified_targets_v15_non_monotonic_grid_shape_bounds_and_speed() -> None:
    hist = DifferentiableHistogram3D(num_bins=16, value_range=(0.0, 1.0))
    x = torch.rand((2, 1, 64, 64, 64), dtype=torch.float32)

    start = time.perf_counter()
    target_hist, targets, guidance_map = generate_unified_targets(
        input_images=x,
        num_bins=16,
        num_chunks=8,
        dark_threshold=0.05,
        hist_module=hist,
        gen_version="v15",
        grid_size=(4, 4, 4),
    )
    elapsed = time.perf_counter() - start

    assert target_hist.shape == (2, 1, 16)
    assert targets.shape == (2, 8)
    assert guidance_map.shape == x.shape
    assert torch.isfinite(guidance_map).all()
    assert torch.all(guidance_map >= 0.0)
    assert torch.all(guidance_map <= 1.0)
    # Keep a loose ceiling to avoid false negatives on slower CI machines.
    assert elapsed < 5.0


def test_generate_unified_targets_v15_strict_background_mask() -> None:
    hist = DifferentiableHistogram3D(num_bins=16, value_range=(0.0, 1.0))
    x = torch.zeros((2, 1, 32, 32, 32), dtype=torch.float32)
    x[:, :, 8:24, 8:24, 8:24] = torch.rand((2, 1, 16, 16, 16)) + 0.05

    _, _, guidance_map = generate_unified_targets(
        input_images=x,
        num_bins=16,
        num_chunks=8,
        dark_threshold=0.05,
        hist_module=hist,
        gen_version="v15",
        grid_size=(4, 4, 4),
    )

    background_mask = x <= 0.01
    assert torch.all(guidance_map[background_mask] == 0.0)


def test_generate_unified_targets_v15_non_monotonic_targets_allow_inversion() -> None:
    hist = DifferentiableHistogram3D(num_bins=16, value_range=(0.0, 1.0))
    torch.manual_seed(123)

    # Build a smooth input ramp so chunk assignments are well-populated.
    ramp = torch.linspace(0.02, 1.0, 32, dtype=torch.float32).view(1, 1, 1, 1, 32)
    x = ramp.expand(1, 1, 32, 32, 32).contiguous()

    _, targets, _ = generate_unified_targets(
        input_images=x,
        num_bins=16,
        num_chunks=8,
        dark_threshold=0.05,
        hist_module=hist,
        gen_version="v15",
        grid_size=(4, 4, 4),
    )

    t = targets[0]
    non_decreasing = bool(torch.all(t[1:] >= t[:-1]))
    non_increasing = bool(torch.all(t[1:] <= t[:-1]))
    assert not non_decreasing
    assert not non_increasing
