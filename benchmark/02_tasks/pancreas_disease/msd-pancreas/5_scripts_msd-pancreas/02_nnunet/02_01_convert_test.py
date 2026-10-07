#!/usr/bin/env python3
"""BIDS (1_BIDS_msd-pancreas/onc-pancreas-msd/) -> flat nnU-Net TEST set 2_nnUNet_msd-pancreas/raw/{imagesTs,labelsTs}_ct/ (item `ct`: MSD Task07 portal-venous CT, a modality the MRI-trained pansegdata models never saw).
Ground truth = UNION of the MSD labels pancreas (1) + cancer (2), binarised to 1: PanSegData's whole-gland mask includes cysts/lesions inside the contour, so the matching target is the gland INCLUDING its lesion.
Exclusions fixed BEFORE predicting (recorded in the manifest): union mask > 250 mL (beyond the largest PanSegData training mask, 175 mL: a bulky mass replaces the gland, not a gland test).
Orientation: LPS via the shared orient.reorient_file on a copy. FOV: the measured per-case crop-before-predict rule shared with totalsegmri-pancreas (see crop_window(); the CT in-plane FOV is up to 500 mm).
Asserts per case: grids equal, labels subset of {0,1,2}, union non-empty, volume unchanged by reorientation (<1%), crop never clips the mask. Case ids carry no contrast suffix."""
from __future__ import annotations
import csv, json, shutil, sys
from pathlib import Path
import nibabel as nib
import numpy as np

DATASET_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = DATASET_ROOT.parents[3]
sys.path.insert(0, str(REPO_ROOT / "benchmark" / "00_commun_scripts" / "00_00_utils"))
from orient import TARGET_AXCODES, reorient_file  # noqa: E402

BIDS = DATASET_ROOT / "1_BIDS_msd-pancreas" / "onc-pancreas-msd"
OUT = DATASET_ROOT / "2_nnUNet_msd-pancreas" / "raw"
ITEMS = {"ct": "portalvenous_CT"}
SOURCE_LABELS_TR = REPO_ROOT / "benchmark/02_tasks/pancreas_disease/pansegdata/2_nnUNet_pansegdata/raw/Dataset150_PanSegData_T1WCE/labelsTr"
MIN_ML = 20.0
MARGIN_MM = 10.0                      # a crop window is always extended to keep the whole pancreas mask + this margin (the lesion/organ is never clipped)


def source_fov_stats():
    """Measured, not assumed: per LPS axis, the SOURCE training images' FOV extent p90/p95 (mm) and the median pancreas-centroid fraction of the FOV."""
    ext, frac = [[], [], []], [[], [], []]
    for lab in sorted(SOURCE_LABELS_TR.glob("*.nii.gz")):
        L = nib.load(lab); a = np.asanyarray(L.dataobj) > 0; z = np.array(L.header.get_zooms()[:3]); e = np.array(L.shape[:3]) * z
        c = np.array([i.mean() for i in np.where(a)]) * z
        for k in range(3): ext[k].append(e[k]); frac[k].append(c[k] / e[k])
    return [dict(p90=float(np.percentile(ext[k], 90)), p95=float(np.percentile(ext[k], 95)), cenfrac=float(np.median(frac[k]))) for k in range(3)]


def crop_window(mask, zooms, shape, st):
    """Per axis: if the test FOV is longer than the source's p95, keep a window of the source's p90 length placed so the pancreas centroid sits at the source's median fraction;
    widen it to contain the whole mask + MARGIN_MM; leave shorter axes alone. Returns voxel slices and a description."""
    ix = np.where(mask); sl, desc = [], []
    for k in range(3):
        ext = shape[k] * zooms[k]
        if ext <= st[k]["p95"]:
            sl.append(slice(0, shape[k])); desc.append(None); continue
        mlo, mhi = ix[k].min() * zooms[k] - MARGIN_MM, (ix[k].max() + 1) * zooms[k] + MARGIN_MM
        W = st[k]["p90"]; lo = float(ix[k].mean() * zooms[k]) - st[k]["cenfrac"] * W; lo = min(max(lo, 0.0), max(ext - W, 0.0)); hi = lo + W
        lo, hi = max(min(lo, mlo), 0.0), min(max(hi, mhi), ext)
        a, b = int(np.floor(lo / zooms[k])), int(np.ceil(hi / zooms[k])); sl.append(slice(a, min(b, shape[k]))); desc.append([a, min(b, shape[k]), round(ext)])
    return tuple(sl), desc


MAX_ML = 250.0


def iter_cases():
    for r in csv.DictReader(open(BIDS / "participants.tsv"), delimiter="\t"):
        sub = r["participant_id"]
        yield (f"msd-pancreas_{sub[3:]}", BIDS / sub / "anat" / f"{sub}_acq-portalvenous_CT.nii.gz",
               BIDS / "derivatives/labels" / sub / "anat" / f"{sub}_acq-portalvenous_label-pancreascancer_dseg.nii.gz", dict(r, item_class="ct"))


def exclude(case_id, meta):
    if float(meta["union_ml"]) > MAX_ML: return f"pancreas+cancer mask {meta['union_ml']} mL > {MAX_ML:g} mL (bulky mass, beyond source masks)"
    return None


def main():
    st = source_fov_stats()
    man = {"cases": {}, "excluded": {}, "rules": {"max_union_ml": MAX_ML, "orientation": "".join(TARGET_AXCODES)},
           "fov_rule": {"source_fov_stats_LPS_axes_mm": st, "margin_mm": MARGIN_MM, "text": "axis longer than source p95 -> window of source p90 length, pancreas centroid at the source median fraction, widened to contain the mask+margin; shorter axes untouched"}}
    for it in ITEMS:
        for p in ("imagesTs", "labelsTs"): (OUT / f"{p}_{it}").mkdir(parents=True, exist_ok=True)
    for cid, img, lab, meta in iter_cases():
        why = exclude(cid, meta)
        if why: man["excluded"][cid] = why; continue
        item = meta["item_class"]
        di, dl = OUT / f"imagesTs_{item}" / f"{cid}_0000.nii.gz", OUT / f"labelsTs_{item}" / f"{cid}.nii.gz"
        shutil.copyfile(img, di)
        Lraw = nib.load(lab); ar = np.asanyarray(Lraw.dataobj); assert set(np.unique(ar)) <= {0, 1, 2}, f"{cid}: unexpected label values"
        nib.save(nib.Nifti1Image((ar > 0).astype(np.uint8), Lraw.affine, Lraw.header), dl)
        raw = "".join(nib.aff2axcodes(nib.load(di).affine))
        vol0 = float((np.asanyarray(nib.load(dl).dataobj) > 0).sum() * np.prod(nib.load(dl).header.get_zooms()[:3]) / 1000)
        reorient_file(di); reorient_file(dl)
        I, L = nib.load(di), nib.load(dl); a = np.asanyarray(L.dataobj)
        assert I.shape == L.shape and np.allclose(I.affine, L.affine, atol=1e-3), f"{cid}: grids differ"
        assert set(np.unique(a)) <= {0, 1} and a.sum() > 0, f"{cid}: bad labels"
        vol1 = float(a.sum() * np.prod(L.header.get_zooms()[:3]) / 1000)
        assert abs(vol1 - vol0) / vol0 < 0.01, f"{cid}: volume changed by reorientation {vol0}->{vol1}"
        sl, crop = crop_window(a > 0, L.header.get_zooms()[:3], L.shape, st)
        if any(c is not None for c in crop):
            I, L = I.slicer[sl], nib.Nifti1Image(a[sl].astype(np.uint8), nib.load(dl).slicer[sl].affine, L.header)
            assert int(a[sl].sum()) == int(a.sum()), f"{cid}: crop clipped the mask"
            nib.save(I, di)
        nib.save(nib.Nifti1Image(np.asanyarray(L.dataobj).astype(np.uint8), L.affine, L.header), dl)
        man["cases"][cid] = dict(item=item, raw_axcodes=raw, axcodes="".join(nib.aff2axcodes(I.affine)), shape=list(I.shape), crop_vox_range_and_orig_ext_mm_per_axis=crop, mask_touches_border=meta["union_touches_border"], cancer_ml=float(meta["cancer_ml"]), zooms=[round(float(z), 3) for z in I.header.get_zooms()[:3]],
                                 pancreas_ml=round(vol1, 1), source_case=meta["source_case"])
    ml = lambda d: sorted(v["pancreas_ml"] for v in d.values())
    q = lambda x: [x[0], x[len(x) // 4], x[len(x) // 2], x[3 * len(x) // 4], x[-1]] if x else []
    src = []
    for f in sorted(SOURCE_LABELS_TR.glob("*.nii.gz")):
        n = nib.load(f); src.append(float((np.asanyarray(n.dataobj) > 0).sum() * np.prod(n.header.get_zooms()[:3]) / 1000))
    man["pancreas_ml_quartiles_min_q1_med_q3_max"] = {"test_all": q(ml(man["cases"])), "source_labelsTr_t1wce": q(sorted(src))}
    for it in ITEMS: man[f"n_{it}"] = sum(1 for v in man["cases"].values() if v["item"] == it)
    (OUT / "msd-pancreas_test_manifest.json").write_text(json.dumps(man, indent=1))
    print("cases:", len(man["cases"]), {it: man[f"n_{it}"] for it in ITEMS}, "excluded:", len(man["excluded"]), "volume quartiles", man["pancreas_ml_quartiles_min_q1_med_q3_max"])


if __name__ == "__main__":
    main()
