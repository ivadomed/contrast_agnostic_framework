#!/usr/bin/env python3
"""BIDS (1_BIDS_totalsegmri-pancreas/abdomen-totalsegmri/) -> flat nnU-Net TEST set 2_nnUNet_totalsegmri-pancreas/raw/{imagesTs,labelsTs}_<item>/ for items t1gre, t2like.
Every exam belongs to exactly ONE item (its acquisition class from participants.tsv, derived in 00_bidsify), so the items hold disjoint cases; case ids carry no contrast suffix.
Exclusions are fixed BEFORE any prediction and written to the manifest with the reason:
  - item_class "other"                         : sequence is neither GRE T1-like nor T2-like organ contrast
  - plane coronal / sagittal                   : the source (PanSegData) is axial only; an in-plane/through-plane mismatch is not what this test is about
  - pancreas mask < 20 mL                      : partial gland (FOV cuts it) or fragment; smallest source training mask is 18 mL, median ~80 mL
Orientation: reoriented to LPS in a copy with the shared orient.reorient_file (raw axcodes recorded). FOV: crop-before-predict (standard of this project) when a test axis is longer than the SOURCE training images allow, see crop_window()/the manifest "fov_rule".
Asserts per case: image/label grids equal, label values subset of {0,1}, non-empty, volume unchanged by reorientation (<1%).
"""
from __future__ import annotations
import csv, json, shutil, sys
from pathlib import Path
import nibabel as nib
import numpy as np

DATASET_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = DATASET_ROOT.parents[3]
sys.path.insert(0, str(REPO_ROOT / "benchmark" / "00_commun_scripts" / "00_00_utils"))
from orient import TARGET_AXCODES, reorient_file  # noqa: E402

BIDS = DATASET_ROOT / "1_BIDS_totalsegmri-pancreas" / "abdomen-totalsegmri"
OUT = DATASET_ROOT / "2_nnUNet_totalsegmri-pancreas" / "raw"
ITEMS = {"t1gre": "gre_T1w", "t2like": "t2like_T2w"}                 # item -> "<acq>_<suffix>" in the BIDS names
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


def iter_cases():
    for r in csv.DictReader(open(BIDS / "participants.tsv"), delimiter="\t"):
        item = r["item_class"]; sub = r["participant_id"]
        if item not in ITEMS:
            yield f"totalsegmri-pancreas_{sub[3:]}", None, None, r; continue
        acq = ITEMS[item]
        yield (f"totalsegmri-pancreas_{sub[3:]}", BIDS / sub / "anat" / f"{sub}_acq-{acq}.nii.gz",
               BIDS / "derivatives/labels" / sub / "anat" / f"{sub}_acq-{acq.split('_')[0]}_label-pancreas_seg.nii.gz", r)


def exclude(case_id, meta):
    if meta["item_class"] not in ITEMS: return f"item_class={meta['item_class']} (not GRE T1-like / T2-like)"
    if meta["plane"] not in ("axial", "3D"): return f"plane={meta['plane']} (source is axial only)"
    if float(meta["pancreas_ml"]) < MIN_ML: return f"pancreas mask {meta['pancreas_ml']} mL < {MIN_ML:g} mL"
    return None


def main():
    st = source_fov_stats()
    man = {"cases": {}, "excluded": {}, "rules": {"min_ml": MIN_ML, "planes": ["axial", "3D"], "orientation": "".join(TARGET_AXCODES)},
           "fov_rule": {"source_fov_stats_LPS_axes_mm": st, "margin_mm": MARGIN_MM, "text": "axis longer than source p95 -> window of source p90 length, pancreas centroid at the source median fraction, widened to contain the mask+margin; shorter axes untouched"}}
    for it in ITEMS:
        for p in ("imagesTs", "labelsTs"): (OUT / f"{p}_{it}").mkdir(parents=True, exist_ok=True)
    for cid, img, lab, meta in iter_cases():
        why = exclude(cid, meta)
        if why: man["excluded"][cid] = why; continue
        item = meta["item_class"]
        di, dl = OUT / f"imagesTs_{item}" / f"{cid}_0000.nii.gz", OUT / f"labelsTs_{item}" / f"{cid}.nii.gz"
        shutil.copyfile(img, di); shutil.copyfile(lab, dl)
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
        man["cases"][cid] = dict(item=item, raw_axcodes=raw, axcodes="".join(nib.aff2axcodes(I.affine)), shape=list(I.shape), crop_vox_range_and_orig_ext_mm_per_axis=crop, mask_touches_border=meta["mask_touches_border"], zooms=[round(float(z), 3) for z in I.header.get_zooms()[:3]],
                                 pancreas_ml=round(vol1, 1), institute=meta["institute"], source=meta["source"], sequence=meta["scanning_sequence"])
    ml = lambda d: sorted(v["pancreas_ml"] for v in d.values())
    q = lambda x: [x[0], x[len(x) // 4], x[len(x) // 2], x[3 * len(x) // 4], x[-1]] if x else []
    src = []
    for f in sorted(SOURCE_LABELS_TR.glob("*.nii.gz")):
        n = nib.load(f); src.append(float((np.asanyarray(n.dataobj) > 0).sum() * np.prod(n.header.get_zooms()[:3]) / 1000))
    man["pancreas_ml_quartiles_min_q1_med_q3_max"] = {"test_all": q(ml(man["cases"])), "source_labelsTr_t1wce": q(sorted(src))}
    for it in ITEMS: man[f"n_{it}"] = sum(1 for v in man["cases"].values() if v["item"] == it)
    (OUT / "totalsegmri-pancreas_test_manifest.json").write_text(json.dumps(man, indent=1))
    print("cases:", len(man["cases"]), {it: man[f"n_{it}"] for it in ITEMS}, "excluded:", len(man["excluded"]), "volume quartiles", man["pancreas_ml_quartiles_min_q1_med_q3_max"])


if __name__ == "__main__":
    main()
