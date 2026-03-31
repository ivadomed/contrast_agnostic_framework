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

CONTRAST_TO_INDEX = {
    "flair": 0,
    "t1w": 1,
    "t1gd": 2,
    "t2w": 3,
}

def normalize_contrast_name(contrast: str) -> str:
    normalized = contrast.strip().lower()
    aliases = {
        "t1": "t1w",
        "t2": "t2w",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in CONTRAST_TO_INDEX:
        valid = ", ".join(sorted(CONTRAST_TO_INDEX))
        raise ValueError(f"Unsupported contrast '{contrast}'. Expected one of: {valid}")
    return normalized


def remap_brats_labels(label):
    """Map BraTS ET label 4 -> 3 while preserving 0/1/2 labels."""
    if isinstance(label, torch.Tensor):
        remapped = label.long()
        return torch.where(remapped == 4, torch.full_like(remapped, 3), remapped)

    remapped = np.asarray(label).copy()
    remapped[remapped == 4] = 3
    return remapped

def get_preprocessing_transforms(
    mode: str = "train",
    patch_size=(128, 128, 128),
    source_contrast: str = "t1w",
):
    """
    Builds the MONAI transform pipeline.
    The images in Decathlon Task01 are 4D (4, H, W, D).
    Channel 0: FLAIR, Channel 1: T1w, Channel 2: T1gd, Channel 3: T2w
    """
    
    source_contrast = normalize_contrast_name(source_contrast)
    source_index = CONTRAST_TO_INDEX[source_contrast]

    # 1. Base transforms for all modes
    transforms_list = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # Resample to isotropic 1x1x1 mm resolution
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        
        # Standardize orientation
        Orientationd(keys=["image", "label"], axcodes="RAS"),

        # Preserve BraTS multiclass labels and map ET id 4 to contiguous id 3.
        Lambdad(keys=["label"], func=remap_brats_labels),
        
        # Extract only the selected source contrast for the single-source framework
        # We keep the shape as (1, H, W, D)
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

    # 4. Spatial cropping with larger patch size support
    transforms_list.append(
        RandSpatialCropd(keys=["image", "label"], roi_size=patch_size, random_size=False)
    )
    
    # 5. Finalize types
    transforms_list.append(EnsureTyped(keys=["image", "label"], data_type="tensor"))
    
    return Compose(transforms_list)

def build_train_dataset(
    data_dir: str,
    patch_size: tuple = (128, 128, 128),
    cache_rate: float = 0.0,
    num_workers: int = 4,
    source_contrast: str = "t1w",
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
    )
    
    dataset = DecathlonDataset(
        root_dir=data_dir,
        task="Task01_BrainTumour", # Tells MONAI exactly which dataset to fetch
        transform=transforms,
        section="training",
        download=True, # Will only download if it doesn't already exist
        cache_rate=cache_rate, # Set higher (e.g., 1.0) if you have lots of RAM to cache data
        num_workers=num_workers # Used by MONAI for parallel data caching
    )
    
    return dataset