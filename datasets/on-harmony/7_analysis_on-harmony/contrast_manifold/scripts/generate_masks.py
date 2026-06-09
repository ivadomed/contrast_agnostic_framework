#!/usr/bin/env python
"""
Phase 2A – Universal Native Brain Macro-Mask Generation

For every NIfTI scan in the ON-Harmony BIDS tree (all modalities, all folders):
  1. Load volume (extract b0 for DWI, vol-0 for BOLD/other 4D)
  2. Otsu threshold
  3. Aggressive binary closing (bridges boundary gaps)
  4. 3D binary fill holes + slice-by-slice fill on all 3 axes (no internal voids)
  5. Final dilation (captures border voxels)
  6. Save uint8 mask to mirrored derivatives/masks/ path

Skips:
  • Files already in derivatives/
  • defacemask files
  • SWI non-magnitude GRE (phase_GRE, imag_GRE, real_GRE) — mag only
  • fmap/epi by default (--include_fmap to override)
  • Existing masks (--force to regenerate)

Usage (run inside tmux):
  tmux new -s native_masks
  set_slot 3 .venv/bin/python analysis/contrast_manifold/scripts/generate_masks.py \
      --bids_dir data/ON-Harmony \
      --n_workers 6
"""

import argparse
import logging
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    generate_binary_structure,
    label as nd_label,
)
from skimage.filters import threshold_otsu
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── BIDS helpers ─────────────────────────────────────────────────────────────

SKIP_SUFFIXES = {"defacemask"}


def parse_bids(filename: str) -> dict:
    stem = filename.replace(".nii.gz", "").replace(".nii", "")
    entities: dict = {}
    for part in stem.split("_"):
        if "-" in part:
            k, v = part.split("-", 1)
            entities[k] = v
        else:
            entities["suffix"] = part
    return entities



def mask_path_for(nii_path: Path, bids_root: Path, deriv_root: Path) -> Path:
    rel = nii_path.relative_to(bids_root)
    mask_name = nii_path.name.replace(".nii.gz", "_mask.nii.gz")
    return deriv_root / "masks" / rel.parent / mask_name


def discover_scans(bids_root: Path) -> list[dict]:
    records = []
    for nii in sorted(bids_root.rglob("*.nii.gz")):
        if "derivatives" in nii.parts:
            continue
        entities = parse_bids(nii.name)
        suffix = entities.get("suffix", "")
        if suffix in SKIP_SUFFIXES:
            continue
        if suffix == "GRE" and not entities.get("part", "").startswith("mag"):
            continue
        records.append({"path": nii, "entities": entities, "suffix": suffix})
    return records


# ─── Volume extraction ────────────────────────────────────────────────────────


def extract_3d(nii_path: Path, entities: dict) -> np.ndarray:
    """
    Load a NIfTI and return a single float32 3D array:
      DWI  → mean of all b0 volumes (b < 100); fallback to vol-0
      BOLD → vol-0 (steady state reference)
      4D other → vol-0
      3D → as-is
    """
    img = nib.load(str(nii_path))
    data = img.get_fdata(dtype=np.float32)

    if data.ndim == 3:
        return data

    if data.ndim == 4:
        suffix = entities.get("suffix", "")
        if suffix == "dwi":
            bval_path = nii_path.with_suffix("").with_suffix(".bval")
            if bval_path.exists():
                bvals = np.loadtxt(bval_path)
                b0_idx = np.where(bvals < 100)[0]
                if b0_idx.size > 0:
                    return data[..., b0_idx].mean(axis=-1)
        # BOLD or any other 4D: take vol-0
        return data[..., 0]

    raise ValueError(f"Unexpected ndim={data.ndim} for {nii_path.name}")


# ─── Mask computation ─────────────────────────────────────────────────────────

# 3-D structuring element: face + edge connectivity (18-connectivity)
_STRUCT = generate_binary_structure(3, 2)


# T2w/FLAIR have dark skulls that sit below the Otsu threshold → need a lower scale.
# All other modalities (T1w, DWI b0, BOLD, GRE mag, EPI) have bright enough tissue at Otsu.
_LOW_THRESH_SUFFIXES = {"T2w", "FLAIR"}


def compute_mask(arr: np.ndarray, suffix: str = "") -> np.ndarray:
    """
    Otsu threshold → aggressive closing → 3D fill + per-axis slice fill → dilation → uint8.
    Guarantees no internal holes in the output mask.
    """
    data = arr.astype(np.float32)

    if data.max() <= 0:
        log.warning("Image is all-zero, returning empty mask")
        return np.zeros(data.shape, dtype=np.uint8)

    scale = 0.5 if suffix in _LOW_THRESH_SUFFIXES else 1.0
    try:
        thresh = threshold_otsu(data) * scale
    except Exception:
        pos = data[data > 0]
        pct = 5 if suffix in _LOW_THRESH_SUFFIXES else 10
        thresh = float(np.percentile(pos, pct)) if pos.size > 0 else 0.0

    binary = data > thresh

    # Close to bridge remaining gaps along the head boundary
    closed = binary_closing(binary, structure=_STRUCT, iterations=5)

    # Keep only the largest connected component (= the head; drops background noise blobs)
    labeled, n_comp = nd_label(closed)
    if n_comp > 1:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0  # ignore background label
        closed = labeled == sizes.argmax()

    # 3D flood-fill from image border
    filled = binary_fill_holes(closed)

    # Slice-by-slice fill on all 3 axes to catch any residual holes
    for ax in range(3):
        for i in range(filled.shape[ax]):
            slc = [slice(None)] * 3
            slc[ax] = i
            filled[tuple(slc)] = binary_fill_holes(filled[tuple(slc)])

    # Expand 2 voxels to ensure border tissue is captured
    dilated = binary_dilation(filled, structure=_STRUCT, iterations=2)
    return dilated.astype(np.uint8)


# ─── Worker ───────────────────────────────────────────────────────────────────


def process_one(nii_path: Path, bids_root: Path, deriv_root: Path, force: bool) -> str:
    """
    Generate and save the native mask for *nii_path*.
    Returns a short status string.
    """
    out_path = mask_path_for(nii_path, bids_root, deriv_root)

    if out_path.exists() and not force:
        return "skip"

    try:
        entities  = parse_bids(nii_path.name)
        vol3d     = extract_3d(nii_path, entities)
        mask      = compute_mask(vol3d, suffix=entities.get("suffix", ""))

        fill_ratio = mask.mean()
        if fill_ratio > 0.95:
            return f"skip_degenerate (fill={fill_ratio:.2f})"

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Preserve the affine/header from the source image
        ref = nib.load(str(nii_path))
        # Use only the spatial part of the header for 3D output
        hdr = ref.header.copy()
        if vol3d.shape != ref.shape[:3]:
            # 4D source: rebuild header for 3D shape
            hdr.set_data_shape(vol3d.shape)
        hdr.set_data_dtype(np.uint8)

        nib.save(nib.Nifti1Image(mask, ref.affine, hdr), str(out_path))
        return "ok"

    except Exception as exc:
        return f"error: {exc}"


# ─── Main ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--bids_dir",        type=Path, default=Path("data/ON-Harmony"))
    p.add_argument("--derivatives_dir", type=Path,
                   default=Path("data/ON-Harmony/derivatives"))
    p.add_argument("--n_workers",  type=int, default=6)
    p.add_argument("--force",      action="store_true",
                   help="Regenerate masks even if they already exist")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log.info("Discovering scans in %s …", args.bids_dir)
    scans = discover_scans(args.bids_dir)
    deriv = args.derivatives_dir
    bids  = args.bids_dir

    if not args.force:
        pending = [s for s in scans
                   if not mask_path_for(s["path"], bids, deriv).exists()]
        log.info("%d total scans | %d already masked | %d to generate",
                 len(scans), len(scans) - len(pending), len(pending))
    else:
        pending = scans
        log.info("%d scans to (re-)generate", len(pending))

    if not pending:
        log.info("Nothing to do.")
        return

    counts = {"ok": 0, "skip": 0, "error": 0}

    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        futures = {
            pool.submit(process_one, s["path"], bids, deriv, args.force): s["path"]
            for s in pending
        }
        with tqdm(as_completed(futures), total=len(futures),
                  desc="Generating masks", unit="file", dynamic_ncols=True) as pbar:
            for fut in pbar:
                path = futures[fut]
                try:
                    status = fut.result()
                except Exception as exc:
                    status = f"error: {exc}"

                if status.startswith("error"):
                    key = "error"
                    pbar.write(f"ERROR {path.name}: {status}")
                elif status.startswith("skip_degenerate"):
                    key = "degenerate"
                    pbar.write(f"SKIP degenerate mask {path.name}: {status}")
                else:
                    key = status
                counts[key] = counts.get(key, 0) + 1

                pbar.set_postfix(ok=counts["ok"], err=counts["error"])

    log.info("Done — OK: %d | Skipped: %d | Degenerate: %d | Errors: %d",
             counts.get("ok", 0), counts.get("skip", 0),
             counts.get("degenerate", 0), counts.get("error", 0))


if __name__ == "__main__":
    main()
