#!/usr/bin/env python3
"""
ToothFairy2 BIDS -> nnU-Net raw (Dataset110_ToothFairy2CBCT).

Thin by design: 00_utils/00_00_extract_and_bidsify.py already did every expensive
and every risky thing (reorient to RAS, resample to 0.6 mm isotropic, reduce the
release's 42 labels to the 3 both cohorts annotate consistently). BIDS is the single source of truth for image content
here — this stage only renames into nnU-Net's layout and writes dataset.json, so
there is exactly one place where voxels are ever touched.

Case ids are `toothfairy2_<NNN>`, derived from the BIDS subject label.

Run via 02_00_convert.sh.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DATASET_ROOT.parent / "00_commun_scripts" / "00_00_utils"))
sys.path.insert(0, str(DATASET_ROOT / "5_scripts_toothfairy2" / "00_utils"))
from nnunet_convert_lib import run_threaded_conversion, write_dataset_json  # noqa: E402
from toothfairy2_labels import NNUNET_LABELS                                 # noqa: E402

BIDS_ROOT = Path(os.environ["BIDS_ROOT"])
NNUNET_RAW = Path(os.environ["nnUNet_raw"])
DS_NAME = os.environ.get("NNUNET_DATASET_ID", "Dataset110_ToothFairy2CBCT")
JOBS = int(os.environ.get("TF2_JOBS", "16"))

OUT = NNUNET_RAW / DS_NAME
IMG_TR, LAB_TR = OUT / "imagesTr", OUT / "labelsTr"
DERIV = BIDS_ROOT / "derivatives" / "labels"


def _case_id(sub: str) -> str:
    return f"toothfairy2_{sub.removeprefix('sub-')}"


def convert(sub: str) -> str:
    cid = _case_id(sub)
    img = BIDS_ROOT / sub / "anat" / f"{sub}_acq-cbct_ct.nii.gz"
    seg = DERIV / sub / "anat" / f"{sub}_acq-cbct_ct_label-maxillofacial_seg.nii.gz"
    for p in (img, seg):
        if not p.exists():
            raise FileNotFoundError(p)
    # Hard-link rather than copy: same filesystem, 480 volumes, and the BIDS tree
    # stays the single source of truth (a copy would silently diverge if BIDS were
    # ever regenerated). Falls back to a copy across filesystems.
    for src, dst in ((img, IMG_TR / f"{cid}_0000.nii.gz"), (seg, LAB_TR / f"{cid}.nii.gz")):
        if dst.exists():
            continue
        try:
            os.link(src, dst)
        except OSError:
            import shutil
            shutil.copy2(src, dst)
    return cid


def main() -> None:
    IMG_TR.mkdir(parents=True, exist_ok=True)
    LAB_TR.mkdir(parents=True, exist_ok=True)
    subs = sorted(p.name for p in BIDS_ROOT.glob("sub-*") if p.is_dir())
    if not subs:
        raise SystemExit(f"no sub-* under {BIDS_ROOT} — run 00_00_extract_and_bidsify.sh first")
    print(f"{len(subs)} subjects -> {OUT}")
    case_ids = run_threaded_conversion(subs, convert, JOBS, progress_every=50)

    write_dataset_json(
        OUT,
        # "CT" is nnU-Net's normalization scheme selector, not a claim about the
        # modality: it triggers the global foreground-intensity z-scoring +
        # percentile clipping that suits a fixed-ish attenuation scale. CBCT has no
        # calibrated Hounsfield scale, but its values are still attenuation-like and
        # far more stable across cases than MRI, so CTNormalization is the right
        # choice (it is also what the ToothFairy2 challenge entries use).
        channel_names={"0": "CT"},
        labels=NNUNET_LABELS,
        num_training=len(case_ids),
        name="ToothFairy2CBCT",
        description="Maxillofacial CBCT, 3 boundary-defined structures "
                    "(mandible, lower teeth, pharynx), 0.6mm iso. Reduced from the "
                    "release's 42 classes to those both cohorts annotate "
                    "consistently — see 00_utils/toothfairy2_labels.py",
        reference="https://ditto.ing.unimore.it/toothfairy2/",
        licence="CC BY-SA 4.0",
        release="20/04/2024 (ToothFairy2 challenge, MICCAI 2024)",
    )
    (OUT / "conversion_summary.json").write_text(json.dumps(
        {"n_cases": len(case_ids), "case_ids": sorted(case_ids)}, indent=2))
    print(f"wrote {len(case_ids)} cases + dataset.json -> {OUT}")


if __name__ == "__main__":
    main()
