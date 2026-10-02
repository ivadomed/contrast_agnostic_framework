#!/usr/bin/env python3
"""
Build acrin6698's TEST-ONLY nnU-Net set -- one test item, `dwi` -- from the
BIDS tree 1_BIDS_acrin6698/onc-breast-acrin6698/ (T0 baseline DWI arm of TCIA
ACRIN-6698, the I-SPY2 DWI sub-study; patients disjoint from our ispy2
training collection, verified 0/385 overlap).

`dwi` = the highest-b trace-weighted volume (b=800 nominal; the b actually used
is recorded per case) -- a contrast neither ispy2 training arm ever saw, so it
is genuinely held-out-contrast (OOD) evidence for BOTH the t1wce- and the
t2w-trained models. Label = ACRIN-6698 "Whole Tumor Manual" DWI ROI on the same
grid (see the BIDS code docstring for its semantics; drawn on DWI/ADC, so it is
reported as its own column, never pooled as if it were the same label as FTV).

Pathology match vs training (same trial, same eligibility -- biopsy-proven
invasive cancer, stage II/III, >=2.5 cm): enforced by T0-only selection at
download (pre-treatment, like every training case). Exclusions, fixed BEFORE
any prediction was run and applied uniformly:
  - subject not converted to BIDS (no T0 DWI mask / download / DICOM failure)
  - DWI-mask side disagrees with the same session's I-SPY2 DCE FTV mask side
    (label inconsistent with the trial's own tumour localisation)
  - lesion crosses the L-R midline (cannot be unilaterally cropped -- duke rule)
Recorded, not filtered: masks spanning <=2 slices, physics ratios (b-high
in/ring, ADC in/ring), FTV-DWI centroid distance.

FOV: ACRIN DWI is axial BILATERAL -> lesion-side half crop on axis 0 after LPS
reorientation (shared unilateral_crop helper; breast task's standard eval FOV).
Case ids: acrin6698_XXXXXX.

Output: 2_nnUNet_acrin6698/raw/{imagesTs,labelsTs}_dwi/ + refB0_dwi/ (b0 skin reference, not a
        test item) + acrin6698_test_manifest.json
Run:    bash 02_nnunet/02_01_convert_test.sh  (run_job, CPU-only)
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

DATASET_ROOT = Path(__file__).resolve().parents[2]            # .../breast_cancer/acrin6698
REPO_ROOT = DATASET_ROOT.parents[3]
sys.path.insert(0, str(REPO_ROOT / "benchmark" / "00_commun_scripts" / "00_00_utils"))
from orient import TARGET_AXCODES, reorient_file  # noqa: E402
from unilateral_crop import crop_axis0, lesion_side_half  # noqa: E402

BIDS = DATASET_ROOT / "1_BIDS_acrin6698" / "onc-breast-acrin6698"
QC_TSV = BIDS / "code" / "logs" / "bidsify_qc.tsv"
OUT = DATASET_ROOT / "2_nnUNet_acrin6698" / "raw"
ITEM = "dwi"


def main() -> None:
    qc = list(csv.DictReader(open(QC_TSV), delimiter="\t"))
    for kind in ("imagesTs", "labelsTs"):
        d = OUT / f"{kind}_{ITEM}"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    if (OUT / "refB0_dwi").exists():
        shutil.rmtree(OUT / "refB0_dwi")
    (OUT / "refB0_dwi").mkdir(parents=True)
    tmp = OUT / "_tmp"
    tmp.mkdir(exist_ok=True)

    cases, excluded = [], []
    raw_ax, b_used = Counter(), Counter()
    for r in qc:
        num = r["source_id"].split("-")[-1]
        cid = f"acrin6698_{num}"
        if r["status"] != "ok":
            excluded.append({"case_id": cid, "reason": r["status"]}); continue
        if r["side_ftv"] != "n/a" and r["side_ftv"] != r["side_dwi"]:
            excluded.append({"case_id": cid, "reason": "dwi_mask_side_disagrees_with_ftv",
                             "centroid_dist_mm": r["ftv_dwi_centroid_dist_mm"]}); continue
        sub = r["participant_id"]
        dwi4 = nib.load(str(BIDS / sub / "dwi" / f"{sub}_acq-trace_dwi.nii.gz"))
        bvals = [int(float(b)) for b in (BIDS / sub / "dwi" / f"{sub}_acq-trace_dwi.bval").read_text().split()]
        hi = int(np.argmax(bvals))
        b_used[bvals[hi]] += 1
        img = nib.Nifti1Image(np.asanyarray(dwi4.dataobj)[..., hi].astype(np.float32), dwi4.affine)
        # b0 reference (same reorient + crop, NOT a test item): fat is dark at high b, so the
        # anterior-skin line used by the A-P crop (02_02) is detected on b0 instead.
        b0 = nib.Nifti1Image(np.asanyarray(dwi4.dataobj)[..., int(np.argmin(bvals))].astype(np.float32), dwi4.affine)
        lab = nib.load(str(BIDS / "derivatives" / "labels" / sub / "dwi" / f"{sub}_acq-trace_label-lesion_seg.nii.gz"))
        raw_ax["".join(nib.aff2axcodes(img.affine))] += 1
        ip, lp, bp = tmp / "img.nii.gz", tmp / "lab.nii.gz", tmp / "b0.nii.gz"
        nib.save(img, str(ip)); nib.save(lab, str(lp)); nib.save(b0, str(bp))
        reorient_file(ip, TARGET_AXCODES); reorient_file(lp, TARGET_AXCODES); reorient_file(bp, TARGET_AXCODES)
        img, lab, b0 = nib.load(str(ip)), nib.load(str(lp)), nib.load(str(bp))
        m = np.asanyarray(lab.dataobj) > 0
        crop = lesion_side_half(m)
        if crop["midline_crossing"]:
            excluded.append({"case_id": cid, "reason": "midline_crossing"}); continue
        img, lab = crop_axis0(img, crop["lo"], crop["hi"]), crop_axis0(lab, crop["lo"], crop["hi"])
        b0 = crop_axis0(b0, crop["lo"], crop["hi"])
        if img.shape != lab.shape or not np.allclose(img.affine, lab.affine, atol=1e-3):
            raise RuntimeError(f"{cid}: image/label grid mismatch")
        nib.save(img, str(OUT / f"imagesTs_{ITEM}" / f"{cid}_0000.nii.gz"))
        nib.save(b0, str(OUT / "refB0_dwi" / f"{cid}_0000.nii.gz"))
        lab_out = nib.Nifti1Image(np.asanyarray(lab.dataobj).astype(np.uint8), lab.affine)
        nib.save(lab_out, str(OUT / f"labelsTs_{ITEM}" / f"{cid}.nii.gz"))
        cases.append({"case_id": cid, "source_id": r["source_id"], "b_value": bvals[hi],
                      "side": crop["side"], "crop_axis0": [crop["lo"], crop["hi"]], "shape": list(lab.shape),
                      "mask_slices": int(r["mask_slices"]), "few_slice_mask": int(r["mask_slices"]) <= 2,
                      "mask_volume_ml": float(r["mask_volume_ml"]),
                      "ratio_bhi_in_vs_ring": r["ratio_bhi_in_vs_ring"], "ratio_adc_in_vs_ring": r["ratio_adc_in_vs_ring"],
                      "ftv_dwi_centroid_dist_mm": r["ftv_dwi_centroid_dist_mm"],
                      "manufacturer": r["manufacturer"], "field_strength": r["field_strength"]})
    shutil.rmtree(tmp)

    def frac(key, pred):
        v = [float(c[key]) for c in cases if c[key] not in ("", "n/a", "None", None)]
        return {"n": len(v), "frac": round(sum(pred(x) for x in v) / max(len(v), 1), 3),
                "median": round(float(np.median(v)), 3) if v else None}
    manifest = {
        "n_cases": len(cases), "item": ITEM, "cases": cases, "excluded": excluded,
        "excluded_counts": dict(Counter(e["reason"].split(":")[0] for e in excluded)),
        "orientation_audit": {"raw_image_axcodes": dict(raw_ax), "target": "".join(TARGET_AXCODES)},
        "qa": {"b_value_used": dict(b_used),
               "bhi_hyperintense_in_mask": frac("ratio_bhi_in_vs_ring", lambda x: x > 1),
               "adc_hypointense_in_mask": frac("ratio_adc_in_vs_ring", lambda x: x < 1),
               "ftv_dwi_centroid_dist_mm": frac("ftv_dwi_centroid_dist_mm", lambda x: x < 20),
               "n_few_slice_masks": sum(c["few_slice_mask"] for c in cases),
               "mask_volume_ml": frac("mask_volume_ml", lambda x: True),
               "manufacturers": dict(Counter(c["manufacturer"] for c in cases))},
        "provenance": {"role": "test-only external evaluation set (models trained on ispy2)",
                       "source": "TCIA ACRIN-6698 (ACRIN 6698/I-SPY2 Breast DWI), T0 baseline study only",
                       "labels": {"background": 0, "tumour": 1},
                       "licence": "CC BY 4.0"}}
    (OUT / "acrin6698_test_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"acrin6698 dwi test set: {len(cases)} cases -> {OUT}")
    print(f"  excluded: {manifest['excluded_counts']}")
    print(f"  qa: {json.dumps(manifest['qa'])}")


if __name__ == "__main__":
    main()
