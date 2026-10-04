#!/usr/bin/env python3
"""FOV / geometry audit: does an eval-only companion's TEST geometry match the TRAINING data's? Run it (CPU job) BEFORE choosing a crop and again after.

Per image it reads the header only (shape, spacing, affine) and reports, per anatomical axis after LPS-style axis assignment (axis 0 = L-R, 1 = A-P, 2 = S-I by
the affine's dominant direction): physical extent in mm and voxel spacing. It compares each test item's distribution to the training set's and flags an axis when the
median extent ratio is outside [1/RATIO, RATIO] (default 1.3) or the spacing ratio is outside [1/SP_RATIO, SP_RATIO] (default 1.5). Full-chest / full-torso test volumes
against organ-/breast-cropped training volumes were the single largest confound found on this project (+10 Dice, -70 mm HD95 once cropped; a fake "largest effect" in a ladder).

  fov_audit.py --train-dir <raw/DatasetXXX/imagesTr> --test-dir <raw/imagesTs_item> [<raw/imagesTs_item2> ...] [--ratio 1.3] [--spacing-ratio 1.5]
Exit code 0 always (it informs the crop decision); prints 'MISMATCH axis=…' lines. Crop with the shared helpers (unilateral_crop.py: lesion_side_half / ap_skin_window /
derive_ap_crop.py for breast; fov.py + 00_02_predict/fov_crop_predict_evaluate.sh for the CHAOS-style S-I slab) or a dataset-local rule documented in the converter.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib


def geometry(path: Path):
    img = nib.load(str(path))
    A = img.affine[:3, :3]
    sp = np.sqrt((A ** 2).sum(0))                       # voxel size per array axis
    dom = np.abs(A / np.maximum(sp, 1e-9)).argmax(0)     # which world axis (0=x L-R, 1=y A-P, 2=z S-I) each array axis runs along
    ext = np.zeros(3); spc = np.zeros(3)
    for ax in range(3):
        w = int(dom[ax]); ext[w] = img.shape[ax] * sp[ax]; spc[w] = sp[ax]
    return ext, spc


def summarize(d: Path, limit=400):
    files = sorted(d.glob("*.nii.gz"))[:limit]
    if not files:
        raise SystemExit(f"no .nii.gz in {d}")
    E, S = zip(*(geometry(f) for f in files))
    return np.array(E), np.array(S), len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True); ap.add_argument("--test-dir", nargs="+", required=True)
    ap.add_argument("--ratio", type=float, default=1.3); ap.add_argument("--spacing-ratio", type=float, default=1.5)
    a = ap.parse_args()
    names = ["L-R", "A-P", "S-I"]
    TE, TS, TN = summarize(Path(a.train_dir))
    print(f"TRAIN {a.train_dir}: n={TN}")
    for i, n in enumerate(names):
        print(f"  {n}: extent mm p5/median/p95 = {np.percentile(TE[:, i], 5):.0f}/{np.median(TE[:, i]):.0f}/{np.percentile(TE[:, i], 95):.0f}   spacing median {np.median(TS[:, i]):.2f}")
    flagged = 0
    for td in a.test_dir:
        E, S, N = summarize(Path(td))
        print(f"TEST {td}: n={N}")
        for i, n in enumerate(names):
            r = np.median(E[:, i]) / max(np.median(TE[:, i]), 1e-9); rs = np.median(S[:, i]) / max(np.median(TS[:, i]), 1e-9)
            tag = ""
            if not (1 / a.ratio <= r <= a.ratio): tag += f"  MISMATCH axis={n} extent ratio {r:.2f} (test/train)"; flagged += 1
            if not (1 / a.spacing_ratio <= rs <= a.spacing_ratio): tag += f"  MISMATCH axis={n} spacing ratio {rs:.2f}"; flagged += 1
            print(f"  {n}: extent mm p5/median/p95 = {np.percentile(E[:, i], 5):.0f}/{np.median(E[:, i]):.0f}/{np.percentile(E[:, i], 95):.0f}   spacing median {np.median(S[:, i]):.2f}{tag}")
    print(f"\nFOV AUDIT: {'NO MISMATCH' if not flagged else f'{flagged} MISMATCH flag(s) -> crop/resample the test items to the training geometry (planned, documented, applied before prediction), then re-audit'}")


if __name__ == "__main__":
    main()
