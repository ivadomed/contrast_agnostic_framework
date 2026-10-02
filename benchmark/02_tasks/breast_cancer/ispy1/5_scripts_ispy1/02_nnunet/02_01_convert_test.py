#!/usr/bin/env python3
"""
Build ispy1's TEST-ONLY nnU-Net set (eval-only cross-dataset companion of the
I-SPY2-trained breast models) from the BIDS tree 1_BIDS_ispy1/onc-breast-ispy1/.

Two test items, same patients, same mask (identical voxel grid -- MAMA-MIA
ships all DCE phases co-registered on one grid), mirroring duke-breast-mri:
  t1wce        <- acq-firstpost_T1w   (same contrast as the t1wce training arm)
  precontrast  <- acq-precontrast_T1w (native T1w; held out for both arms)
Case ids are identical across items (ispy1_XXXX, no contrast suffix) so the
patient-level significance merge pairs them correctly.

Pathology match vs. the I-SPY2 training population (biopsy-proven invasive
cancer, stage II/III, >=2.5 cm, pre-neoadjuvant, primary-tumour mask) --
fixed BEFORE any prediction was run, applied uniformly:
  - I-SPY1 eligibility (TCIA): "T3 tumors measuring at least 3 cm ... receiving
    neoadjuvant chemotherapy"; MAMA-MIA keeps only pre-treatment scans -> same
    disease stage/timepoint as training, no extra filter needed for that.
  - EXCLUDE bilateral breast cancer (MAMA-MIA bilateral_breast_cancer=1): two
    primaries, only one annotated -> an unlabeled real tumour in the image.
  - EXCLUDE breast implants (has_implant=1): not representative anatomy.
  - Size is REPORTED, not filtered: baseline MRI longest diameter (TCIA sheet)
    and mask volume / bbox extent are written to the manifest and summarised
    against I-SPY2's own Dataset100 labelsTr distribution.

FOV: I-SPY1 is sagittal and ~98% unilateral (MAMA-MIA Table 2). Any case whose
L-R physical extent after LPS reorientation exceeds BILATERAL_LR_MM is cropped
to the lesion-side half with the shared unilateral_crop helper (duke's rule,
midline-crossing lesions excluded) -- unilateral is the breast task's only
reported eval FOV.

Output: 2_nnUNet_ispy1/raw/{imagesTs,labelsTs}_{t1wce,precontrast}/ +
        2_nnUNet_ispy1/raw/ispy1_test_manifest.json
Run:    bash 02_nnunet/02_01_convert_test.sh  (run_job, CPU-only)
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

DATASET_ROOT = Path(__file__).resolve().parents[2]            # .../breast_cancer/ispy1
REPO_ROOT = DATASET_ROOT.parents[3]
sys.path.insert(0, str(REPO_ROOT / "benchmark" / "00_commun_scripts" / "00_00_utils"))
from orient import TARGET_AXCODES, reorient_file  # noqa: E402
from unilateral_crop import crop_axis0, lesion_side_half  # noqa: E402

BIDS = DATASET_ROOT / "1_BIDS_ispy1" / "onc-breast-ispy1"
OUT = DATASET_ROOT / "2_nnUNet_ispy1" / "raw"
ISPY2_LABELS_TR = (DATASET_ROOT.parent / "ispy2" / "2_nnUNet_ispy2" / "raw"
                   / "Dataset100_ISPY2T1wce" / "labelsTr")
ITEMS = {"t1wce": "acq-firstpost_T1w", "precontrast": "acq-precontrast_T1w"}
BILATERAL_LR_MM = 250.0


def mask_size(lab: nib.Nifti1Image) -> dict:
    m = np.asanyarray(lab.dataobj) > 0
    z = np.array(lab.header.get_zooms()[:3], dtype=float)
    if not m.any():
        return {"volume_ml": 0.0, "bbox_max_mm": 0.0}
    idx = np.argwhere(m)
    ext = (idx.max(0) - idx.min(0) + 1) * z
    return {"volume_ml": round(float(m.sum() * z.prod() / 1000.0), 2), "bbox_max_mm": round(float(ext.max()), 1)}


def summary(vals: list[float]) -> dict:
    a = np.array(vals, dtype=float)
    return {"n": int(a.size), "p10": round(float(np.percentile(a, 10)), 1),
            "median": round(float(np.median(a)), 1), "p90": round(float(np.percentile(a, 90)), 1)}


def main() -> None:
    parts = pd.read_csv(BIDS / "participants.tsv", sep="\t").set_index("participant_id")
    for item in ITEMS:
        for kind in ("imagesTs", "labelsTs"):
            d = OUT / f"{kind}_{item}"
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

    cases, excluded = [], []
    raw_ax = Counter()
    n_enh_fail = 0
    for sub in sorted(parts.index):
        num = sub.replace("sub-ispy1", "")
        cid = f"ispy1_{num}"
        p = parts.loc[sub]
        if str(p.bilateral_breast_cancer) == "1":
            excluded.append({"case_id": cid, "reason": "bilateral_breast_cancer"}); continue
        if str(p.has_implant) == "1":
            excluded.append({"case_id": cid, "reason": "breast_implant"}); continue
        lab_src = BIDS / "derivatives" / "labels" / sub / "anat" / f"{sub}_acq-firstpost_T1w_label-lesion_seg.nii.gz"
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            lab_tmp = td / "lab.nii.gz"
            shutil.copyfile(lab_src, lab_tmp)
            raw_ax["".join(nib.aff2axcodes(nib.load(str(lab_tmp)).affine))] += 1
            reorient_file(lab_tmp, TARGET_AXCODES)
            lab = nib.load(str(lab_tmp))
            m = np.asanyarray(lab.dataobj)
            vals = set(np.unique(m).astype(int).tolist())
            if not vals <= {0, 1}:
                raise RuntimeError(f"{cid}: non-binary label {sorted(vals)}")
            if not (m > 0).any():
                excluded.append({"case_id": cid, "reason": "empty_mask"}); continue
            lr_mm = lab.shape[0] * float(lab.header.get_zooms()[0])
            crop = None
            if lr_mm > BILATERAL_LR_MM:
                crop = lesion_side_half(m > 0)
                if crop["midline_crossing"]:
                    excluded.append({"case_id": cid, "reason": "bilateral_fov_midline_crossing"}); continue
                lab = crop_axis0(lab, crop["lo"], crop["hi"])
            nib.save(lab, str(OUT / "labelsTs_t1wce" / f"{cid}.nii.gz"))
            shutil.copyfile(OUT / "labelsTs_t1wce" / f"{cid}.nii.gz", OUT / "labelsTs_precontrast" / f"{cid}.nii.gz")
            means = {}
            for item, suffix in ITEMS.items():
                img_tmp = td / f"{item}.nii.gz"
                shutil.copyfile(BIDS / sub / "anat" / f"{sub}_{suffix}.nii.gz", img_tmp)
                reorient_file(img_tmp, TARGET_AXCODES)
                img = nib.load(str(img_tmp))
                if crop:
                    img = crop_axis0(img, crop["lo"], crop["hi"])
                if img.shape != lab.shape or not np.allclose(img.affine, lab.affine, atol=1e-3):
                    raise RuntimeError(f"{cid}/{item}: image/label grid mismatch {img.shape} vs {lab.shape}")
                nib.save(img, str(OUT / f"imagesTs_{item}" / f"{cid}_0000.nii.gz"))
                means[item] = float(np.asanyarray(img.dataobj)[np.asanyarray(lab.dataobj) > 0].mean())
            enh_ok = means["t1wce"] > means["precontrast"]
            n_enh_fail += 0 if enh_ok else 1
            cases.append({"case_id": cid, "source_id": f"ISPY1_{num}", **mask_size(lab),
                          "mri_ld_baseline_mm": None if str(p.mri_ld_baseline_mm) == "n/a" else float(p.mri_ld_baseline_mm),
                          "lr_extent_mm": round(lr_mm, 1), "bilateral_fov_cropped": crop is not None,
                          "crop": crop, "shape": list(lab.shape),
                          "zooms": [round(float(z), 3) for z in lab.header.get_zooms()[:3]],
                          "enhancement_check_post_gt_pre": bool(enh_ok)})

    ref = [mask_size(nib.load(str(f))) for f in sorted(ISPY2_LABELS_TR.glob("*.nii.gz"))]
    manifest = {
        "n_cases": len(cases), "items": list(ITEMS), "cases": cases, "excluded": excluded,
        "orientation_audit": {"raw_label_axcodes": dict(raw_ax), "target": "".join(TARGET_AXCODES)},
        "qa": {"n_enhancement_check_failed": n_enh_fail,
               "n_bilateral_fov_cropped": sum(c["bilateral_fov_cropped"] for c in cases)},
        "size_vs_training": {
            "ispy1_volume_ml": summary([c["volume_ml"] for c in cases]),
            "ispy1_bbox_max_mm": summary([c["bbox_max_mm"] for c in cases]),
            "ispy1_mri_ld_baseline_mm": summary([c["mri_ld_baseline_mm"] for c in cases if c["mri_ld_baseline_mm"] is not None]),
            "ispy2_trainpool_volume_ml": summary([r["volume_ml"] for r in ref]),
            "ispy2_trainpool_bbox_max_mm": summary([r["bbox_max_mm"] for r in ref])},
        "provenance": {
            "role": "test-only external evaluation set (models trained on ispy2)",
            "source": "MAMA-MIA (Garrucho et al. 2025, Synapse syn60868042), I-SPY1 subset; images = "
                      "TCIA ISPY1 (ACRIN 6657) DCE phases _0000 (pre) / _0001 (1st post)",
            "labels": {"background": 0, "tumour": 1},
            "label_semantics": "MAMA-MIA expert primary-lesion mask on 1st post-contrast phase "
                               "(multifocal: primary only; clips excluded; necrosis included) -- the "
                               "same protocol as duke-breast-mri's masks",
            "licence": "CC BY 3.0 (TCIA ISPY1; MAMA-MIA README lists ISPY1 as CC BY 3.0)"}}
    (OUT / "ispy1_test_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"ispy1 test set: {len(cases)} cases -> {OUT}")
    print(f"  excluded: {Counter(e['reason'] for e in excluded)}")
    print(f"  raw label axcodes: {dict(raw_ax)}  enhancement failures: {n_enh_fail}  "
          f"bilateral-FOV cropped: {manifest['qa']['n_bilateral_fov_cropped']}")
    print(f"  size vs training: {json.dumps(manifest['size_vs_training'])}")


if __name__ == "__main__":
    main()
