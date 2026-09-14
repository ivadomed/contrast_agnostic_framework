#!/usr/bin/env python3
"""
Build the nnUNet test-input dirs for I-SPY2 cross-dataset evaluation of ambl-trained
models. I-SPY2 is EVALUATION-ONLY (see 00_utils/env.sh): no native training here.

CRITICAL FILTER (2026-09-04): many I-SPY2 cases are UNILATERAL (single-breast FOV),
a fundamentally different acquisition/FOV protocol from ambl's always-bilateral
training population -- conflating that with the cross-contrast shift this dataset is
meant to isolate would be wrong. This script reads
../../4_splits_ispy2/unilateral_fov_exclusions.json (the screening pass's source of
truth) and builds test inputs ONLY for the 122 usable bilateral cases -- every case
in that file's excluded_unilateral_cases / no_bids_data lists is skipped outright,
never just filtered out downstream.

Two items (mirrors ambl's own t1wce/t2w training-modality split so 05_predict/
06_evaluate stay parallel):
    t1wce -> sub-ispy2<N>_acq-firstpost_T1w.nii.gz   (matches ambl's T1WCE channel)
    t2w   -> sub-ispy2<N>_T2w.nii.gz

Reads:  ../../1_BIDS_ispy2/breast-ispy2/sub-ispy2<N>/anat/  (+ derivatives/labels masks)
Writes: ../../2_nnUNet_ispy2/raw/imagesTs_<item>/{case}_0000.nii.gz
                                 /labelsTs_<item>/{case}.nii.gz

Case id = participant label without the sub- prefix (e.g. ispy2105286).

Verifies nib.aff2axcodes for EVERY case/item against the expected ('L','P','S')
orientation (matches ambl's own training data, confirmed separately) before copying
-- this project has been bitten twice by silently skipping this check (see CLAUDE.md
"Verify orientation on onboarding").

    .venv/bin/python 05_00_build_test_inputs.py
"""
import gzip
import json
import shutil
import sys
from pathlib import Path

import nibabel as nib

DATASET_ROOT = Path(__file__).resolve().parents[2]
BIDS_ROOT     = DATASET_ROOT / "1_BIDS_ispy2" / "breast-ispy2"
DERIV_DIR     = BIDS_ROOT / "derivatives" / "labels"
NNUNET_RAW    = DATASET_ROOT / "2_nnUNet_ispy2" / "raw"
EXCLUSIONS_FP = DATASET_ROOT / "4_splits_ispy2" / "unilateral_fov_exclusions.json"

EXPECTED_AXCODES = ("L", "P", "S")

ITEMS = {
    "t1wce": "acq-firstpost_T1w",   # matches ambl's T1WCE channel
    "t2w":   "T2w",
}


def gzip_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".gz":
        shutil.copyfile(src, dst)
    else:
        with open(src, "rb") as f_in, gzip.open(dst, "wb", compresslevel=1) as f_out:
            shutil.copyfileobj(f_in, f_out)


def usable_cases() -> list[str]:
    """Derive the 122 usable bilateral cases: all subs with BIDS data, minus every
    pid in excluded_unilateral_cases and no_bids_data. Cross-checked against the
    file's own `counts.usable_bilateral` so a schema drift fails loudly, not silently."""
    d = json.loads(EXCLUSIONS_FP.read_text())
    excluded = {e["pid"] for e in d["excluded_unilateral_cases"]}
    no_bids = {e["pid"] for e in d["no_bids_data"]}
    all_subs = sorted(p.name for p in BIDS_ROOT.glob("sub-ispy2*") if p.is_dir())

    def sub2pid(sub: str) -> str:
        return "ISPY2-" + sub[len("sub-ispy2"):]

    usable = [s for s in all_subs if sub2pid(s) not in excluded and sub2pid(s) not in no_bids]
    expected = d["counts"]["usable_bilateral"]
    if len(usable) != expected:
        raise SystemExit(
            f"Derived {len(usable)} usable cases but exclusions file's own counts say "
            f"{expected} -- schema drift or a bad assumption in sub2pid(). Aborting."
        )
    return usable


def main() -> None:
    if not EXCLUSIONS_FP.exists():
        raise SystemExit(f"Exclusion list not found: {EXCLUSIONS_FP}")
    subs = usable_cases()
    print(f"[build_test_inputs] {len(subs)} usable bilateral cases (verified against "
          f"{EXCLUSIONS_FP.name}'s own counts)")

    orientation_bad: list[str] = []

    for item, suffix in ITEMS.items():
        img_dir = NNUNET_RAW / f"imagesTs_{item}"
        lab_dir = NNUNET_RAW / f"labelsTs_{item}"
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)

        n_ok, missing = 0, []
        for sub in subs:
            case = sub[len("sub-"):]                       # ispy2105286
            img = BIDS_ROOT / sub / "anat" / f"{sub}_{suffix}.nii.gz"
            seg = DERIV_DIR / sub / "anat" / f"{sub}_{suffix}_label-lesion_seg.nii.gz"
            if not img.exists() or not seg.exists():
                missing.append(case)
                continue

            ax_img = nib.aff2axcodes(nib.load(img).affine)
            ax_seg = nib.aff2axcodes(nib.load(seg).affine)
            if ax_img != EXPECTED_AXCODES or ax_seg != EXPECTED_AXCODES:
                orientation_bad.append(f"{case}/{item}: img={ax_img} seg={ax_seg}")
                continue

            gzip_copy(img, img_dir / f"{case}_0000.nii.gz")
            gzip_copy(seg, lab_dir / f"{case}.nii.gz")
            n_ok += 1

        status = f"{n_ok}/{len(subs)} -> {img_dir.name} (+labels)"
        if missing:
            status += f"  MISSING: {missing}"
        print(status)

    if orientation_bad:
        print("\n[build_test_inputs] ORIENTATION MISMATCH (expected "
              f"{EXPECTED_AXCODES}), refusing to trust these cases:", file=sys.stderr)
        for line in orientation_bad:
            print(f"  ! {line}", file=sys.stderr)
        raise SystemExit(1)

    print(f"[build_test_inputs] all cases confirmed {EXPECTED_AXCODES} — "
          "matches ambl's own training-data orientation.")


if __name__ == "__main__":
    main()
