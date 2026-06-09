import os
import numpy as np
import torch
from monai.apps import DecathlonDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    SpatialPadd,
    RandAffined,
    Rand3DElasticd,             # <-- ADDED IMPORT
    RandSimulateLowResolutiond,
    RandGaussianNoised,
    RandGaussianSmoothd,
    ScaleIntensityd,
    RandSpatialCropd,
    Lambdad,
    EnsureTyped
)

from src.training.data_registry import get_dataset_spec


DEFAULT_DATASET_NAME = "brats2017"
DEFAULT_CONTRASTS = tuple(get_dataset_spec(DEFAULT_DATASET_NAME)["contrasts"])


def build_contrast_to_index(contrasts: list[str] | tuple[str, ...] | None) -> dict[str, int]:
    ordered = list(contrasts) if contrasts else list(DEFAULT_CONTRASTS)
    return {str(name).strip().lower(): idx for idx, name in enumerate(ordered)}


def normalize_contrast_name(
    contrast: str,
    available_contrasts: list[str] | tuple[str, ...] | None = None,
) -> str:
    normalized = contrast.strip().lower()
    aliases = {
        "t1": "t1w",
        "t2": "t2w",
    }
    normalized = aliases.get(normalized, normalized)
    contrast_to_index = build_contrast_to_index(available_contrasts)
    if normalized not in contrast_to_index:
        valid = ", ".join(sorted(contrast_to_index))
        raise ValueError(f"Unsupported contrast '{contrast}'. Expected one of: {valid}")
    return normalized


def remap_labels(label, label_mapping: dict[int, int] | None = None):
    if not label_mapping:
        return label

    mapping = {int(k): int(v) for k, v in label_mapping.items()}
    if isinstance(label, torch.Tensor):
        remapped = label.long()
        for src, dst in mapping.items():
            remapped = torch.where(remapped == src, torch.full_like(remapped, dst), remapped)
        return remapped

    remapped = np.asarray(label).copy()
    for src, dst in mapping.items():
        remapped[remapped == src] = dst
    return remapped

def get_preprocessing_transforms(
    mode: str = "train",
    patch_size=(128, 128, 128),
    source_contrast: str = "t1w",
    contrasts: list[str] | tuple[str, ...] | None = None,
    label_mapping: dict[int, int] | None = None,
    label_free: bool = False,
):
    """
    Builds the MONAI transform pipeline.
    The images in Decathlon Task01 are 4D (4, H, W, D).
    Channel 0: FLAIR, Channel 1: T1w, Channel 2: T1gd, Channel 3: T2w
    """
    
    if not contrasts:
        raise ValueError("Expected non-empty 'contrasts' from cfg.data.contrasts.")
    contrast_to_index = build_contrast_to_index(contrasts)
    source_contrast = normalize_contrast_name(source_contrast, list(contrast_to_index.keys()))
    source_index = contrast_to_index[source_contrast]

    # 1. Base transforms for all modes
    if label_free:
        transforms_list = [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            Orientationd(keys=["image"], axcodes="RAS"),
            Lambdad(keys=["image"], func=lambda x: x[source_index : source_index + 1, ...]),
        ]
    else:
        transforms_list = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Lambdad(keys=["label"], func=lambda y: remap_labels(y, label_mapping=label_mapping)),
            Lambdad(keys=["image"], func=lambda x: x[source_index : source_index + 1, ...]),
        ]
    
    # 2. Add robust train-only augmentations before intensity normalization
    if mode in ("train", "train_bigaug", "train_lpci"):
        transforms_list.extend(
            [
                # We migrated the following CPU augmentations to GPU in src/kornia_augmentations.py:
                # - RandSimulateLowResolutiond
                # - RandGaussianNoised
                # - RandGaussianSmoothd
            ]
        )

    # 3. Normalize intensities to [0, 1] after train-time augmentations
    transforms_list.append(ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0))

    spatial_keys = ["image"] if label_free else ["image", "label"]

    transforms_list.append(SpatialPadd(keys=spatial_keys, spatial_size=patch_size))
    transforms_list.append(RandSpatialCropd(keys=spatial_keys, roi_size=patch_size, random_size=False))
    transforms_list.append(EnsureTyped(keys=spatial_keys, data_type="tensor"))
    
    return Compose(transforms_list)

def build_train_dataset(
    data_dir: str,
    task_name: str,
    patch_size: tuple = (128, 128, 128),
    cache_rate: float = 0.0,
    num_workers: int = 4,
    source_contrast: str = "t1w",
    contrasts: list[str] | tuple[str, ...] | None = None,
    label_mapping: dict[int, int] | None = None,
):
    """
    Automatically downloads and loads the Decathlon Brain Tumour dataset.
    Matches the exact signature expected by train.py.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    transforms = get_preprocessing_transforms(
        mode="train",
        patch_size=patch_size,
        source_contrast=source_contrast,
        contrasts=contrasts,
        label_mapping=label_mapping,
    )
    
    dataset = DecathlonDataset(
        root_dir=data_dir,
        task=task_name,
        transform=transforms,
        section="training",
        download=True, # Will only download if it doesn't already exist
        cache_rate=cache_rate, # Set higher (e.g., 1.0) if you have lots of RAM to cache data
        num_workers=num_workers # Used by MONAI for parallel data caching
    )
    
    return dataset