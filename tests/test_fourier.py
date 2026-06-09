from __future__ import annotations

import sys
from pathlib import Path

import torch

# Setup project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.augmentation.kornia_augmentations import RandomFourierAmplitude3D


def test_random_fourier_amplitude_3d_properties() -> None:
    torch.manual_seed(0)
    x = torch.rand(2, 1, 8, 8, 8, dtype=torch.float32, requires_grad=True)

    aug = RandomFourierAmplitude3D(
        p=1.0,
        low_freq_ratio=0.2,
        scale_range=(0.5, 1.5),
    )

    y = aug(x)

    assert y.shape == x.shape
    assert torch.is_floating_point(y)
    assert not torch.is_complex(y)
    assert torch.all(y >= 0.0)
    assert torch.all(y <= 1.0)
    assert y.requires_grad

    # Ensure graph connectivity is preserved for backward.
    y.sum().backward()
    assert x.grad is not None
