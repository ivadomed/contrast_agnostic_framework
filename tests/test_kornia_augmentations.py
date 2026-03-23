from __future__ import annotations

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Setup project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.kornia_augmentations import RandomElasticTransform3D, build_kornia_augmentation


def _base_cfg(*, affine_prob: float, elastic_prob: float) -> object:
    return OmegaConf.create(
        {
            "training": {
                "generator": {
                    "gpu_aug": {
                        "affine_prob": affine_prob,
                        "affine_rotate_range": [0.0, 0.0, 0.0],
                        "affine_scale_range": [0.0, 0.0, 0.0],
                        "elastic_prob": elastic_prob,
                        "elastic_sigma_range": [4.0, 4.0],
                        "elastic_magnitude_range": [2.0, 2.0],
                    }
                }
            }
        }
    )


def test_random_elastic_transform_identity_when_disabled() -> None:
    x = torch.rand(2, 1, 16, 16, 16, dtype=torch.float32)
    aug = RandomElasticTransform3D(
        p=0.0,
        sigma_range=(4.0, 4.0),
        magnitude_range=(2.0, 2.0),
    )

    y = aug(x)

    assert y.shape == x.shape
    assert torch.equal(y, x)


def test_random_elastic_transform_preserves_batch_shape() -> None:
    torch.manual_seed(0)
    x = torch.rand(3, 1, 16, 16, 16, dtype=torch.float32)
    aug = RandomElasticTransform3D(
        p=1.0,
        sigma_range=(4.0, 4.0),
        magnitude_range=(2.0, 2.0),
    )

    y = aug(x)

    assert y.shape == x.shape
    # With p=1 and non-zero magnitude the output should not be identically equal.
    assert not torch.equal(y, x)


def test_kornia_pipeline_identity_when_all_probs_zero() -> None:
    x = torch.rand(2, 1, 16, 16, 16, dtype=torch.float32)
    cfg = _base_cfg(affine_prob=0.0, elastic_prob=0.0)
    aug = build_kornia_augmentation(cfg)
    aug.low_res.p = 0.0
    aug.noise.p = 0.0
    aug.smooth.p = 0.0

    y = aug(x)

    assert y.shape == x.shape
    assert torch.equal(y, x)


def test_kornia_pipeline_output_bounds_and_shape() -> None:
    torch.manual_seed(123)
    x = torch.rand(2, 1, 16, 16, 16, dtype=torch.float32)
    cfg = _base_cfg(affine_prob=1.0, elastic_prob=1.0)
    aug = build_kornia_augmentation(cfg)

    y = aug(x)

    assert y.shape == x.shape
    assert torch.all(y >= 0.0)
    assert torch.all(y <= 1.0)
