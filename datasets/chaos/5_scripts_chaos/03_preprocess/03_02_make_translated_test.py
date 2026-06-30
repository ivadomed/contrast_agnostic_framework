#!/usr/bin/env python3
"""
Build TRANSLATED copies of the chaos nnUNet test sets for a translation-robustness
experiment (translation-equivariance test): shift each test IMAGE and its GT MASK
identically by a fixed fraction of the volume along the left–right (L) axis, in
VOXEL/ARRAY space, then run the normal chaos models on them.

Why array-space (not the affine): nnU-Net resamples by spacing and predicts in the
input voxel grid, so an affine-only origin shift is a no-op (the model sees the same
array). To actually perturb the input we must move the array CONTENT. Image and mask
are shifted by the identical integer voxel shift, so the organ moves with its label
and Dice measures "does the model still segment the organ wherever it now sits."

Shift mechanics:
  * fraction f (default 0.5) of the L-axis size, rounded to an integer voxel count.
  * direction "left" = toward anatomical L. The L axis and its polarity are read per
    volume from the affine (nib.aff2axcodes), so this is correct regardless of how a
    given volume is stored.
  * vacated voxels are filled with background: image → its own minimum intensity,
    label → 0. Content pushed past the FOV edge is dropped (a real 50% shift removes
    half the anatomy — intentional, that is the stress test).

Reads / writes (per dataset, per modality), under 2_nnUNet_chaos/raw/<DS>/:
  imagesTs_<mod>/<case>_0000.nii.gz  → imagesTs_<mod>_translation_<NNN>/<case>_0000.nii.gz
  labelsTs_<mod>/<case>.nii.gz       → labelsTs_<mod>_translation_<NNN>/<case>.nii.gz
where <NNN> = round(f*100) (e.g. 050). The affine/header is preserved (only the array
content moves), so predictions land on the same grid as the shifted GT.

Usage (via 03_02_make_translated_test.sh → run_job):
  python 03_02_make_translated_test.py [--frac 0.5] [--datasets Dataset060... Dataset061...]
                                       [--mods t1in t1out t2spir ct] [--direction left]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

DATASET_ROOT = Path(__file__).resolve().parents[2]          # …/datasets/chaos/
NNUNET_RAW   = DATASET_ROOT / "2_nnUNet_chaos" / "raw"
DEF_DATASETS = ["Dataset060_CHAOS_MR_T1in", "Dataset061_CHAOS_MR_T2spir"]
DEF_MODS     = ["t1in", "t1out", "t2spir", "ct"]


def lr_axis_and_sign(affine, direction: str):
    """Return (array_axis, shift_sign) so that shifting the array by sign*N along axis
    moves content toward `direction` ('left'→anatomical L, 'right'→R).

    aff2axcodes gives the anatomical code each array axis increases toward, e.g.
    ('L','P','S') means axis0 increases toward L. If axis a increases toward L, then
    a positive index shift moves content toward L."""
    codes = nib.aff2axcodes(affine)
    want_pos = "L" if direction == "left" else "R"   # which code means "toward direction"
    for ax, c in enumerate(codes):
        if c in ("L", "R"):
            # +shift moves content toward higher index = toward code `c`.
            sign = +1 if c == want_pos else -1
            return ax, sign
    raise ValueError(f"no L/R axis found in axcodes {codes}")


def shift_array(arr: np.ndarray, axis: int, n: int, fill) -> np.ndarray:
    """Shift `arr` by `n` voxels along `axis` (n may be negative), filling vacated
    voxels with `fill`. Content moved past the edge is dropped (no wrap)."""
    out = np.full_like(arr, fill)
    if n == 0:
        return arr.copy()
    src = [slice(None)] * arr.ndim
    dst = [slice(None)] * arr.ndim
    if n > 0:
        dst[axis] = slice(n, None); src[axis] = slice(0, arr.shape[axis] - n)
    else:
        dst[axis] = slice(0, arr.shape[axis] + n); src[axis] = slice(-n, None)
    out[tuple(dst)] = arr[tuple(src)]
    return out


def translate_file(src: Path, dst: Path, frac: float, direction: str, is_label: bool) -> str:
    im = nib.load(str(src))
    arr = np.asanyarray(im.dataobj)
    ax, sign = lr_axis_and_sign(im.affine, direction)
    n = sign * int(round(frac * arr.shape[ax]))
    fill = 0 if is_label else arr.min()
    out = shift_array(arr, ax, n, fill)
    dst.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(out.astype(arr.dtype), im.affine, im.header), str(dst))
    return f"axis{ax} shift {n:+d}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frac", type=float, default=0.5, help="shift fraction of L-axis size")
    ap.add_argument("--direction", choices=["left", "right"], default="left")
    ap.add_argument("--datasets", nargs="+", default=DEF_DATASETS)
    ap.add_argument("--mods", nargs="+", default=DEF_MODS)
    args = ap.parse_args()
    tag = f"translation_{round(args.frac * 100):03d}"

    print("=" * 64)
    print(f"chaos translated test sets → *_{tag}  (shift {args.direction} {args.frac:.0%} of L axis)")
    print("=" * 64)
    total = 0
    for ds in args.datasets:
        for mod in args.mods:
            img_src = NNUNET_RAW / ds / f"imagesTs_{mod}"
            lab_src = NNUNET_RAW / ds / f"labelsTs_{mod}"
            if not img_src.is_dir():
                print(f"  skip {ds}/{mod}: no imagesTs_{mod}"); continue
            img_dst = NNUNET_RAW / ds / f"imagesTs_{mod}_{tag}"
            lab_dst = NNUNET_RAW / ds / f"labelsTs_{mod}_{tag}"
            n_ok = 0; info = ""
            for f in sorted(img_src.glob("*.nii.gz")):
                info = translate_file(f, img_dst / f.name, args.frac, args.direction, is_label=False)
                n_ok += 1
            for f in sorted(lab_src.glob("*.nii.gz")):
                translate_file(f, lab_dst / f.name, args.frac, args.direction, is_label=True)
            # copy dataset.json sidecar if the predict step expects it (chaos predictions carry one)
            print(f"  {ds}/{mod}: {n_ok} images + labels → *_{tag}  [{info}]")
            total += n_ok
    if total == 0:
        sys.exit("ERROR: nothing translated — check --datasets/--mods and that imagesTs_* exist.")
    print(f"  done: {total} image+label pairs translated.")


if __name__ == "__main__":
    main()
