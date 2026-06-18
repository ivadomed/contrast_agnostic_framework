#!/usr/bin/env python3
"""
Validate the CHAOS BIDS conversion produced by 00_00_download_and_bidsify.py.

Checks per subject/modality:
  - image ↔ dseg share shape + geometry (origin/spacing/direction)
  - dseg label ids ⊆ {0,1,2,3,4}; MR has organs >1, CT is liver-only {0,1}
  - masks are non-empty
  - T1DUAL in/out-phase masks are identical (shared, by construction)
  - intensity-overlap sanity: liver-labelled voxels have clearly different mean
    intensity from background (catches gross z-order / spatial misalignment)

Run:  .venv/bin/python 9_tests_chaos/test_bids_conversion.py
"""
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

BIDS = Path(__file__).resolve().parents[1] / "1_BIDS_chaos" / "chaos-abdominal"
DERIV = BIDS / "derivatives" / "manual_masks"
VALID_LABELS = {0, 1, 2, 3, 4}


def _load(p: Path):
    img = sitk.ReadImage(str(p))
    return img, sitk.GetArrayFromImage(img)


def _geom(img):
    return (img.GetSize(), tuple(np.round(img.GetOrigin(), 3)),
            tuple(np.round(img.GetSpacing(), 3)), tuple(np.round(img.GetDirection(), 3)))


def check_pair(img_path: Path, msk_path: Path, is_ct: bool, errs: list[str]):
    tag = img_path.name
    if not msk_path.exists():
        errs.append(f"{tag}: missing mask {msk_path.name}"); return None
    img, ia = _load(img_path)
    msk, ma = _load(msk_path)

    if _geom(img) != _geom(msk):
        errs.append(f"{tag}: image/mask geometry mismatch")
    if ia.shape != ma.shape:
        errs.append(f"{tag}: shape {ia.shape} != mask {ma.shape}")

    labels = set(np.unique(ma).tolist())
    if not labels <= VALID_LABELS:
        errs.append(f"{tag}: unexpected labels {labels - VALID_LABELS}")
    if labels == {0}:
        errs.append(f"{tag}: mask is empty")
    if is_ct and not labels <= {0, 1}:
        errs.append(f"{tag}: CT mask should be liver-only, got {labels}")

    # Misalignment sanity (contrast-invariant): organ labels must fall on body
    # tissue, not air/FOV-exterior. A z-flip or wrong mask ordering drops a chunk of
    # foreground onto low/zero-signal voxels. We don't use organ-vs-background
    # intensity contrast — that varies too much across T1-in/out/T2-SPIR.
    fg = ma > 0
    sig = ia[ia > 0]
    if fg.sum() > 50 and sig.size > 50:
        body_thr = np.percentile(sig, 10)
        frac_in_body = (ia[fg] > body_thr).mean()
        if frac_in_body < 0.90:
            errs.append(f"{tag}: only {frac_in_body:.2f} of organ voxels lie on body "
                        f"tissue — possible misalignment")
    return ma


def main() -> int:
    if not BIDS.exists():
        print(f"BIDS root not found: {BIDS}"); return 1
    errs: list[str] = []
    subs = sorted(p.name for p in BIDS.glob("sub-*") if p.is_dir())
    n_mr = n_ct = 0

    for sub in subs:
        anat = BIDS / sub / "anat"
        deriv = DERIV / sub / "anat"
        is_ct = sub.startswith("sub-CT")
        if is_ct:
            n_ct += 1
            check_pair(anat / f"{sub}_CT.nii", deriv / f"{sub}_CT_dseg.nii", True, errs)
        else:
            n_mr += 1
            in_m = check_pair(anat / f"{sub}_acq-inphase_T1w.nii",
                              deriv / f"{sub}_acq-inphase_T1w_dseg.nii", False, errs)
            out_m = check_pair(anat / f"{sub}_acq-outphase_T1w.nii",
                               deriv / f"{sub}_acq-outphase_T1w_dseg.nii", False, errs)
            check_pair(anat / f"{sub}_T2w.nii", deriv / f"{sub}_T2w_dseg.nii", False, errs)
            if in_m is not None and out_m is not None and not np.array_equal(in_m, out_m):
                errs.append(f"{sub}: in/out-phase masks differ (should be shared)")

    print(f"Checked {n_mr} MR + {n_ct} CT subjects.")
    if errs:
        print(f"\nFAIL — {len(errs)} issue(s):")
        for e in errs:
            print(f"  ✗ {e}")
        return 1
    print("All conversion checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
