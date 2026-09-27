#!/usr/bin/env python3
"""
Build the nnUNet test-input dir for the SLIVER07 evaluation set.

SLIVER07 is CT-only (the single "contrast" available). The chaos models are
single-channel, so each CT volume is fed as channel _0000. This materialises an
image dir + matching GT label dir so 06_evaluate can score predictions.

Reads:  ../../1_BIDS_sliver07/liver-sliver07-ct/sub-LV##/anat/  (+ derivatives masks)
Writes: ../../2_nnUNet_sliver07/raw/imagesTs_ct/{case}_0000.nii.gz
                                    /labelsTs_ct/{case}.nii.gz

Case id = participant label without the sub- prefix (e.g. LV01). The "_ct" suffix
mirrors chaos's per-modality test dirs so 05_predict / 06_evaluate stay parallel.

BIDS files are already .nii.gz (compressed) — plain_copy() below just copies them
as-is (NOT a gzip_copy — the BIDS source used to be uncompressed .nii, which this
function used to gzip on the fly; now that it's already compressed, doing that again
would double-gzip the output into an unreadable file — a real bug caught 2026-08-28,
same class of bug found in chaos's equivalent script).

    python 05_00_build_test_inputs.py
"""
import shutil
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]
BIDS_ROOT    = DATASET_ROOT / "1_BIDS_sliver07" / "liver-sliver07-ct"
DERIV_DIR    = BIDS_ROOT / "derivatives" / "labels"
NNUNET_RAW   = DATASET_ROOT / "2_nnUNet_sliver07" / "raw"

MODALITY = "ct"   # the only contrast in SLIVER07


def plain_copy(src: Path, dst: Path) -> None:
    """Copy an already-compressed .nii.gz file as-is. No-op if dst exists."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    subs = sorted(p.name for p in BIDS_ROOT.glob("sub-LV*") if p.is_dir())
    if not subs:
        raise SystemExit(f"No sub-LV* under {BIDS_ROOT} — run 00_00_download_and_bidsify.py first.")

    img_dir = NNUNET_RAW / f"imagesTs_{MODALITY}"
    lab_dir = NNUNET_RAW / f"labelsTs_{MODALITY}"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    n_ok, missing = 0, []
    for sub in subs:
        case = sub[len("sub-"):]                       # LV01
        img = BIDS_ROOT / sub / "anat" / f"{sub}_CT.nii.gz"
        seg = DERIV_DIR / sub / "anat" / f"{sub}_CT_label-liver_seg.nii.gz"
        if not img.exists() or not seg.exists():
            missing.append(case)
            continue
        plain_copy(img, img_dir / f"{case}_0000.nii.gz")
        plain_copy(seg, lab_dir / f"{case}.nii.gz")
        n_ok += 1

    status = f"{n_ok}/{len(subs)} → {img_dir.name} (+labels)"
    if missing:
        status += f"  [MISSING {len(missing)}: {missing[:5]}]"
    print(f"  {MODALITY:7s}: {status}")


if __name__ == "__main__":
    main()
