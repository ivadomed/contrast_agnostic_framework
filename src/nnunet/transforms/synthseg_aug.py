"""
On-the-fly SynthSeg synthesis via BrainGenerator.

SynthSeg-A and SynthSeg-B generate ENTIRELY synthetic (image, label) pairs from
BrainGenerator — the real T1w images from the data loader are NOT used as input.
The BrainGenerator samples a random label map from its labels_dir (which contains
ONLY training-fold subjects' SynthSeg masks, enforcing subject isolation), then
generates a synthetic MRI image from it.

The output image is normalized to z-score (brain ≈ N(0,1), background = 0) to
match the nnUNet ZScoreNormalization convention expected by the rest of the pipeline.

BrainGenerator is TensorFlow/Keras and runs on CPU.  It is initialized once at
training start and called synchronously inside generate_train_batch().
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import nibabel as nib
import numpy as np
import torch

# FreeSurfer label ID → 7-class mapping (mirrors 01_convert_dataset.py)
# For 31-class: set env var NNUNET_LABEL_SET=31class
FREESURFER_TO_7CLASS: Dict[int, int] = {
    0:  0, 2:  2, 3:  1, 4:  3, 5:  3, 7:  6, 8:  6,
    10: 4, 11: 4, 12: 4, 13: 4, 14: 3, 15: 3, 16: 5,
    17: 4, 18: 4, 26: 4, 28: 4, 41: 2, 42: 1, 43: 3,
    44: 3, 46: 6, 47: 6, 49: 4, 50: 4, 51: 4, 52: 4,
    53: 4, 54: 4, 58: 4, 60: 4,
}

_FS_IDS_31 = [2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,26,28,41,42,43,44,46,47,49,50,51,52,53,54,58,60]
FREESURFER_TO_31CLASS: Dict[int, int] = {0: 0, **{fs: i+1 for i, fs in enumerate(_FS_IDS_31)}}

# Select active mapping from env var (set in train.sh / config.sh)
import os as _os
_LABEL_SET = _os.environ.get("NNUNET_LABEL_SET", "7class")
_ACTIVE_MAP = FREESURFER_TO_31CLASS if _LABEL_SET == "31class" else FREESURFER_TO_7CLASS

_MAX_LABEL = max(_ACTIVE_MAP.keys())
_REMAP_LUT  = np.zeros(_MAX_LABEL + 2, dtype=np.uint8)
for fs_id, cls in _ACTIVE_MAP.items():
    _REMAP_LUT[fs_id] = cls


def _remap_labels(arr: np.ndarray) -> np.ndarray:
    """Vectorised FreeSurfer → N-class remap using a lookup table (7class or 31class)."""
    clipped = np.clip(arr, 0, _MAX_LABEL + 1).astype(np.int32)
    return _REMAP_LUT[clipped].astype(np.uint8)


def _normalize_to_zscore(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Z-score normalize image within brain tissue (labels > 0).
    Background (labels == 0) is set to exactly 0.
    """
    brain_mask = labels > 0
    out = np.zeros_like(image, dtype=np.float32)
    if brain_mask.any():
        brain_vals = image[brain_mask].astype(np.float64)
        mu  = brain_vals.mean()
        std = brain_vals.std() + 1e-7
        out[brain_mask] = ((image[brain_mask] - mu) / std).astype(np.float32)
    return out


def _pad_crop_to_shape(data: np.ndarray, target: tuple) -> np.ndarray:
    """Centre-pad or centre-crop a 3D array to target shape."""
    out = np.zeros(target, dtype=data.dtype)
    src_sl, dst_sl = [], []
    for s, t in zip(data.shape, target):
        if s <= t:
            pad = (t - s) // 2
            src_sl.append(slice(None))
            dst_sl.append(slice(pad, pad + s))
        else:
            crop = (s - t) // 2
            src_sl.append(slice(crop, crop + t))
            dst_sl.append(slice(None))
    out[tuple(dst_sl)] = data[tuple(src_sl)]
    return out


def _prepare_fixed_shape_labels(labels_dir: Path, target_shape: tuple) -> Path:
    """Pad/crop all label maps in labels_dir to target_shape, write to a temp dir."""
    tmp = Path(tempfile.mkdtemp(prefix="synthseg_fixedshape_"))
    for p in sorted(labels_dir.glob("*.nii.gz")):
        img = nib.load(str(p))
        data = np.asarray(img.dataobj, dtype=np.int32)
        fixed = _pad_crop_to_shape(data, target_shape)
        nib.save(nib.Nifti1Image(fixed, img.affine, img.header), str(tmp / p.name))
    return tmp


def build_brain_generator(labels_dir: Path, mode: str = "A", target_shape: tuple = None):
    """
    Instantiate a BrainGenerator for SynthSeg mode A or B.

    Mode A — dense label sampling (uniform priors, independent per class):
      prior_distributions='uniform'
      mix_prior_and_random=False

    Mode B — EM mixture (uses prior statistics of real data):
      prior_distributions='uniform'
      mix_prior_and_random=True   (blends prior with random draws)

    Parameters
    ----------
    labels_dir : Path
        Directory containing SynthSeg mask .nii.gz symlinks (training-fold only).
    mode : "A" or "B"
    """
    project_root = Path(__file__).resolve().parents[3]
    synthseg_path = str(project_root / "SynthSeg")
    if synthseg_path not in sys.path:
        sys.path.insert(0, synthseg_path)

    # Import TF and restrict to CPU before SynthSeg triggers CUDA init.
    # TF_USE_LEGACY_KERAS=1 is set in the training shell env (_train_method.sh).
    import tensorflow as tf  # noqa: PLC0415
    tf.config.set_visible_devices([], "GPU")

    from SynthSeg.brain_generator import BrainGenerator  # noqa: PLC0415

    # BrainGenerator builds one TF model keyed to the first label's shape.
    # Pre-pad/crop all maps to a fixed size so every subject feeds the same model.
    # target_shape must be >= initial_patch_size so crops in generate_synthseg_batch
    # don't extend past the volume and pad with zeros.
    if target_shape is None:
        target_shape = (224, 288, 192)  # safe default > any expected initial_patch_size
    fixed_labels_dir = _prepare_fixed_shape_labels(labels_dir, tuple(target_shape))

    if mode == "A":
        bg = BrainGenerator(
            labels_dir=str(fixed_labels_dir),
            prior_distributions="uniform",
            mix_prior_and_random=False,
            flipping=True,
            randomise_res=False,
            bias_field_std=0.7,
        )
    elif mode == "B":
        bg = BrainGenerator(
            labels_dir=str(fixed_labels_dir),
            prior_distributions="uniform",
            mix_prior_and_random=True,
            flipping=True,
            randomise_res=False,
            bias_field_std=0.7,
        )
    else:
        raise ValueError(f"Unknown SynthSeg mode: {mode!r}. Use 'A' or 'B'.")

    return bg


def _random_crop(image_z: np.ndarray, labels_7: np.ndarray, patch_size: Tuple[int, int, int]) -> Dict[str, torch.Tensor]:
    cd, ch, cw = patch_size
    D, H, W = image_z.shape
    d0 = np.random.randint(0, max(1, D - cd + 1))
    h0 = np.random.randint(0, max(1, H - ch + 1))
    w0 = np.random.randint(0, max(1, W - cw + 1))
    d1, h1, w1 = min(d0 + cd, D), min(h0 + ch, H), min(w0 + cw, W)
    d0, h0, w0 = d1 - cd, h1 - ch, w1 - cw
    patch     = image_z[d0:d1, h0:h1, w0:w1][np.newaxis, np.newaxis]
    patch_seg = labels_7[d0:d1, h0:h1, w0:w1][np.newaxis, np.newaxis]
    return {
        "data":   torch.from_numpy(patch).float(),
        "target": torch.from_numpy(patch_seg.astype(np.int16)),
        "keys":   ["synthseg_generated"],
    }


def generate_synthseg_batch(
    brain_generator,
    initial_patch_size: Tuple[int, int, int],
    n_crops: int = 1,
) -> "list[Dict[str, torch.Tensor]]":
    """
    Call BrainGenerator once, normalize, and return n_crops random-crop batches.

    Returning n_crops crops from one brain call amortizes the ~2s/call cost.
    Each item in the returned list has:
      'data'   : (1, 1, cd, ch, cw) float32 z-score image
      'target' : (1, 1, cd, ch, cw) int16 7-class labels
    """
    image_np, labels_np = brain_generator.generate_brain()

    if image_np.ndim == 4:
        image_np = image_np[0]
    if labels_np.ndim == 4:
        labels_np = labels_np[0]

    # BrainGenerator (nibabel) gives axes in (x,y,z) = (sagittal, coronal, axial).
    # nnUNet preprocessed data (SimpleITK GetArrayFromImage) is in (z,y,x) = (axial, coronal, sagittal).
    # Transpose (2,1,0) aligns axis convention. Inconsistent flips in train viz are from
    # BrainGenerator's flipping=True augmentation — intentional, not a bug.
    image_np  = np.ascontiguousarray(image_np.transpose(2, 1, 0))
    labels_np = np.ascontiguousarray(labels_np.transpose(2, 1, 0))

    labels_7 = _remap_labels(labels_np.astype(np.int32))
    image_z  = _normalize_to_zscore(image_np.astype(np.float32), labels_7)

    return [_random_crop(image_z, labels_7, initial_patch_size) for _ in range(n_crops)]
