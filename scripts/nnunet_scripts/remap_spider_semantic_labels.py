#!/usr/bin/env python3
"""Remap SPIDER instance-style labels to semantic classes expected by Dataset102 json.

Mapping:
- 0            -> 0 (background)
- 1..99        -> 1 (vertebra)
- 100          -> 3 (canal)
- 200..299     -> 2 (disc)
- everything else -> 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--labels-dir",
        action="append",
        required=True,
        help="labelsTr directory to remap (can be passed multiple times)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print stats without writing")
    return p.parse_args()


def remap(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.int16, copy=False)
    y = np.zeros_like(x, dtype=np.uint8)
    y[(x >= 1) & (x <= 99)] = 1
    y[(x >= 200) & (x <= 299)] = 2
    y[x == 100] = 3
    return y


def main() -> None:
    args = parse_args()
    total = 0
    for d in args.labels_dir:
        labels_dir = Path(d)
        files = sorted(labels_dir.glob("*.nii.gz"))
        print(f"[remap] {labels_dir} -> {len(files)} files")
        for f in files:
            nii = nib.load(str(f))
            src = np.asarray(nii.dataobj)
            dst = remap(src)
            if not args.dry_run:
                out = nib.Nifti1Image(dst, nii.affine, nii.header)
                nib.save(out, str(f))
            total += 1
    print(f"[remap] processed {total} files")


if __name__ == "__main__":
    main()
