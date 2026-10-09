#!/usr/bin/env python3
"""BIDS -> nnU-Net raw for BOTH isles2022 training contrasts (single input channel each).

  Dataset140_ISLES2022_DWI / Dataset141_ISLES2022_FLAIR / each:
    imagesTr/<id>_0000.nii.gz  labelsTr/<id>.nii.gz        train pool (200)
    imagesTs_{dwi,flair}/<id>_0000.nii.gz  labelsTs_{dwi,flair}/<id>.nii.gz   held-out test (47)
    dataset.json
Everything is on the DWI/mask grid (LAS). DWI is copied unchanged; ADC (in the release, derived from DWI) is deliberately NOT used (Paul, 2026-10-04); FLAIR (own native grid in all
250 cases) is resampled onto the DWI grid (trilinear, zero outside its FOV) -- one mask serves all contrasts.
Per-case QC -> isles2022_convert_manifest.json: FLAIR coverage of the lesion (fraction of lesion voxels inside
the FLAIR FOV with FLAIR>0) and lesion/brain ratios per contrast (physics check: DWI>1, FLAIR>1).
Fails if any case has FLAIR lesion coverage < 0.95 (FOV cut the lesion)."""
from __future__ import annotations
import json, shutil, sys
from pathlib import Path
import numpy as np, nibabel as nib
from nibabel.processing import resample_from_to

DS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DS.parents[2] / "00_commun_scripts" / "00_00_utils"))
from nnunet_convert_lib import run_threaded_conversion, write_dataset_json  # noqa: E402

BIDS = DS / "1_BIDS_isles2022" / "stroke-brain-isles2022"
SPL = DS / "4_splits_isles2022"
RAW = DS / "2_nnUNet_isles2022" / "raw"
DATASETS = {"dwi": "Dataset140_ISLES2022_DWI", "flair": "Dataset141_ISLES2022_FLAIR"}
TESTS = ("dwi", "flair")

def save(arr, ref, dst, dtype):
    dst.parent.mkdir(parents=True, exist_ok=True)
    o = nib.Nifti1Image(arr.astype(dtype), ref.affine); o.set_qform(ref.affine, 1); o.set_sform(ref.affine, 1)
    o.set_data_dtype(dtype); nib.save(o, str(dst))

def paths(cid):
    s = "sub-strokecase" + cid.split("_")[1]; b = f"{s}_ses-0001"
    return {"dwi": BIDS/s/"ses-0001/dwi"/f"{b}_dwi.nii.gz", "flair": BIDS/s/"ses-0001/anat"/f"{b}_FLAIR.nii.gz", "msk": BIDS/"derivatives/labels"/s/"ses-0001/dwi"/f"{b}_label-lesion_seg.nii.gz"}

def convert(item):
    cid, split = item
    p = paths(cid)
    msk = nib.load(str(p["msk"])); m = np.asanyarray(msk.dataobj).astype(np.uint8)
    assert set(np.unique(m)) <= {0, 1} and m.any(), cid
    img = {"dwi": nib.load(str(p["dwi"]))}
    for k in img:
        assert img[k].shape == msk.shape and np.allclose(img[k].affine, msk.affine, atol=1e-3), (cid, k)
    fl = resample_from_to(nib.load(str(p["flair"])), (msk.shape, msk.affine), order=1, cval=0.0)
    arrs = {"dwi": np.asanyarray(img["dwi"].dataobj).astype(np.float32),
            "flair": np.asanyarray(fl.dataobj).astype(np.float32)}
    les = m > 0
    qc = {"case": cid, "split": split, "shape": list(msk.shape), "spacing": [round(float(z), 3) for z in msk.header.get_zooms()[:3]],
          "lesion_ml": round(float(les.sum() * np.prod(msk.header.get_zooms()[:3]) / 1000), 3),
          "flair_lesion_cov": round(float((arrs["flair"][les] > 0).mean()), 4)}
    for k, a in arrs.items():
        brain = (arrs["dwi"] > 0) & ~les
        qc[f"{k}_les_over_brain"] = round(float(a[les].mean() / max(a[brain].mean(), 1e-6)), 3)
    for ds_id, ds in DATASETS.items():
        out = RAW / ds
        if split == "train":
            save(arrs[ds_id], msk, out / "imagesTr" / f"{cid}_0000.nii.gz", np.float32)
            save(m, msk, out / "labelsTr" / f"{cid}.nii.gz", np.uint8)
        else:
            for t in TESTS:
                save(arrs[t], msk, out / f"imagesTs_{t}" / f"{cid}_0000.nii.gz", np.float32)
                save(m, msk, out / f"labelsTs_{t}" / f"{cid}.nii.gz", np.uint8)
    QC.append(qc)
    return cid

QC: list = []
def main():
    part = json.loads((SPL / "partition.json").read_text())
    items = [(c, "train") for c in part["train_pool"]] + [(c, "test") for c in part["test"]]
    run_threaded_conversion(items, convert, jobs=4, progress_every=25)
    for ds_id, ds in DATASETS.items():
        write_dataset_json(RAW / ds, {"0": ds_id.upper()}, {"background": 0, "lesion": 1}, len(part["train_pool"]),
                           name=f"ISLES2022_{ds_id.upper()}", description="ISLES'22 acute/subacute ischemic stroke lesion segmentation",
                           licence="CC BY 4.0 + source terms (no redistribution without ISLES'22 team written agreement)",
                           reference="https://doi.org/10.1038/s41597-022-01875-5")
    QC.sort(key=lambda q: q["case"])
    (DS / "2_nnUNet_isles2022" / "isles2022_convert_manifest.json").write_text(json.dumps(QC, indent=1))
    bad = [q["case"] for q in QC if q["flair_lesion_cov"] < 0.95]
    for k in ("dwi", "flair"):
        x = np.array([q[f"{k}_les_over_brain"] for q in QC]); print(k, "lesion/brain median", np.median(x), "frac>1", (x > 1).mean())
    print("n", len(QC), "FLAIR lesion coverage min", min(q["flair_lesion_cov"] for q in QC), "cases <0.95:", bad)
    if bad: raise SystemExit(f"FLAIR FOV cuts the lesion in {bad}")

if __name__ == "__main__":
    main()
