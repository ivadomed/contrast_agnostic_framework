#!/usr/bin/env python3
"""BIDS -> nnU-Net raw for BOTH PanSegData training contrasts (single input channel each).

  Dataset150_PanSegData_T1WCE / Dataset151_PanSegData_T2W / each:
    imagesTr/<id>_0000.nii.gz  labelsTr/<id>.nii.gz        train pool (own contrast image + the mask drawn on it)
    imagesTs_{t1wce,t2w}/<id>_0000.nii.gz  labelsTs_{t1wce,t2w}/<id>.nii.gz   held-out test: EVERY test subject in BOTH items,
                                                                              each scored against the mask drawn on that scan
    dataset.json
Case ids 'pansegdata_<site><NNNN>' carry no contrast suffix (the patient-level significance merge pairs the two items on them).
Every volume is reoriented to the project's canonical LPS with the shared 00_00_utils/orient.reorient_file (lossless permute/flip, no resampling);
the BIDS leaf stores mixed LPI/LPS/RAS. The BIDS labels already sit on their image's grid (the 66 mismatching label headers were fixed there);
this script asserts it per scan. Native, non-uniform spacing is kept (nnU-Net resamples at preprocessing).
Per-scan QC -> pansegdata_convert_manifest.json: raw axcodes, shape, spacing, mask volume, physics contrast (mask vs ring)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, nibabel as nib

DS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DS.parents[2] / "00_commun_scripts" / "00_00_utils"))
from nnunet_convert_lib import run_threaded_conversion, write_dataset_json  # noqa: E402
from orient import reorient_file  # noqa: E402

BIDS = DS / "1_BIDS_pansegdata" / "onc-pancreas-pansegdata"
SPL = DS / "4_splits_pansegdata"
RAW = DS / "2_nnUNet_pansegdata" / "raw"
DATASETS = {"t1wce": "Dataset150_PanSegData_T1WCE", "t2w": "Dataset151_PanSegData_T2W"}
ITEMS = ("t1wce", "t2w")
STEM = {"t1wce": ("{sub}_acq-venous", "T1w"), "t2w": ("{sub}", "T2w")}
QC: list = []


def paths(cid, item):
    sub = "sub-" + cid.split("pansegdata_")[1]
    stem, suf = STEM[item]; stem = stem.format(sub=sub)
    return BIDS / sub / "anat" / f"{stem}_{suf}.nii.gz", BIDS / "derivatives/labels" / sub / "anat" / f"{stem}_label-pancreas_seg.nii.gz"


def write_pair(item, cid, img_dir, lab_dir, ref_img, ref_lab, arr, m):
    di, dl = img_dir / f"{cid}_0000.nii.gz", lab_dir / f"{cid}.nii.gz"
    di.parent.mkdir(parents=True, exist_ok=True); dl.parent.mkdir(parents=True, exist_ok=True)
    o = nib.Nifti1Image(arr.astype(np.float32), ref_img.affine); o.set_qform(ref_img.affine, 1); o.set_sform(ref_img.affine, 1); o.set_data_dtype(np.float32)
    nib.save(o, str(di))
    l = nib.Nifti1Image(m.astype(np.uint8), ref_img.affine); l.set_qform(ref_img.affine, 1); l.set_sform(ref_img.affine, 1); l.set_data_dtype(np.uint8)
    nib.save(l, str(dl))
    for p in (di, dl):
        reorient_file(p)


def convert(item_):
    cid, split = item_
    cached = {}
    for it in ITEMS:
        pi, pl = paths(cid, it)
        img, lab = nib.load(str(pi)), nib.load(str(pl))
        assert img.shape == lab.shape and np.allclose(img.affine, lab.affine, atol=1e-3), (cid, it, "label not on image grid")
        x = np.asanyarray(img.dataobj).astype(np.float32); m = (np.asanyarray(lab.dataobj) > 0)
        assert set(np.unique(m)) <= {False, True} and m.any(), (cid, it)
        z = img.header.get_zooms()[:3]
        QC.append({"case": cid, "item": it, "split": split, "raw_axcodes": "".join(nib.aff2axcodes(img.affine)), "shape": list(img.shape),
                   "spacing": [round(float(v), 3) for v in z], "mask_ml": round(float(m.sum() * np.prod(z) / 1000), 2),
                   "mask_over_all_mean": round(float(x[m].mean() / max(x[x > 0].mean(), 1e-6)), 3)})
        cached[it] = (img, lab, x, m)
    for ds_item, ds in DATASETS.items():
        out = RAW / ds
        if split == "train":
            img, lab, x, m = cached[ds_item]
            write_pair(ds_item, cid, out / "imagesTr", out / "labelsTr", img, lab, x, m)
        else:
            for t in ITEMS:
                img, lab, x, m = cached[t]
                write_pair(t, cid, out / f"imagesTs_{t}", out / f"labelsTs_{t}", img, lab, x, m)
    return cid


def main():
    part = json.loads((SPL / "partition.json").read_text())
    items = [(c, "train") for c in part["train_pool"]] + [(c, "test") for c in part["test"]]
    run_threaded_conversion(items, convert, jobs=4, progress_every=25)
    for ds_item, ds in DATASETS.items():
        write_dataset_json(RAW / ds, {"0": ds_item.upper()}, {"background": 0, "pancreas": 1}, len(part["train_pool"]),
                           name=f"PanSegData_{ds_item.upper()}", description="PanSegData multi-center pancreas MRI (pancreas segmentation)",
                           licence="CC BY-NC 4.0", reference="https://doi.org/10.1016/j.media.2024.103382")
    QC.sort(key=lambda q: (q["case"], q["item"]))
    (DS / "2_nnUNet_pansegdata" / "pansegdata_convert_manifest.json").write_text(json.dumps(QC, indent=1))
    print("cases", len(items), "scans converted", len(QC), "| raw axcodes seen:", sorted({q["raw_axcodes"] for q in QC}))
    for it in ITEMS:
        v = np.array([q["mask_over_all_mean"] for q in QC if q["item"] == it])
        print(it, "pancreas/tissue mean-intensity ratio median", round(float(np.median(v)), 3), "frac>1", round(float((v > 1).mean()), 3))


if __name__ == "__main__":
    main()
