#!/usr/bin/env python
"""PyRadiomics extraction, sharded + resumable (one CSV append per unit).
  --task brats  : patients = cross_dataset_fillswap_model.brats_patients() (<=40); rows patient x contrast x region
  --task xds    : non-BraTS dev source keys (cases <= --maxc), rows key x case x label; native z-score, resampled to 1 mm iso
  --task openms : open-ms keys, ONLY allowed after outputs/data/radiomics_xds_frozen.json exists (blind-stage guard)
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from radiomics_common import DATA, MIN_VOX, PAD_MM, make_extractors, region_row  # noqa: E402
import cross_dataset_fillswap_model as xd  # noqa: E402  (import only; its extract() is never called)
import nibabel as nib  # noqa: E402

_EX = None
MAX_LABELS = 6


def _init():
    global _EX
    _EX = make_extractors()


def brats_unit(pid):
    tex, shp = _EX
    from scipy.ndimage import distance_transform_edt  # noqa: F401
    rows = []
    try:
        lbl = np.squeeze(np.asarray(nib.load(str(xd.LABDIR / f"{pid}.nii.gz")).dataobj)).round().astype(int)
        raw = {c: np.squeeze(np.asarray(nib.load(str(xd.BIDS / f"sub-{pid}" / "anat" / f"sub-{pid}_{s}")).dataobj)).astype(np.float32)
               for c, s in xd.BSUF.items()}
        brain = np.zeros(lbl.shape, bool)
        for a in raw.values():
            brain |= a != 0
        brain |= lbl > 0
        for c, a in raw.items():
            v = a[brain]
            z = np.where(brain, (a - v.mean()) / max(v.std(), 1e-6), 0).astype(np.float32)
            for rn, rid in xd.BREG.items():
                R = lbl == rid
                if R.sum() < MIN_VOX:
                    continue
                idx = np.argwhere(R)
                lo = np.maximum(idx.min(0) - PAD_MM, 0)
                hi = np.minimum(idx.max(0) + PAD_MM + 1, R.shape)
                cs = tuple(slice(a_, b_) for a_, b_ in zip(lo, hi))
                f = region_row(tex, shp, z[cs], R[cs], brain[cs])
                if f is None:
                    continue
                rows.append({"patient": pid, "contrast": c, "region": rn, **f})
    except Exception as e:  # noqa: BLE001
        print(f"SKIP {pid}: {e!r}", flush=True)
    return pid, rows


def xds_unit(task):
    import SimpleITK as sitk
    tex, shp = _EX
    key, cid, ip, lp, mand = task
    rows = []
    try:
        im = nib.load(str(ip))
        img = np.squeeze(np.asarray(im.dataobj)).astype(np.float32)
        while img.ndim > 3:
            img = img[..., 0]
        zooms = np.asarray(im.header.get_zooms()[:3], float)
        z, _, fg, _ = xd.prep(img, zooms)
        lbl = np.squeeze(np.asarray(nib.load(str(lp)).dataobj)).round().astype(int)
        labs = [1] if mand else [int(v) for v in np.unique(lbl) if v > 0]
        if len(labs) > MAX_LABELS:  # many-label parcellations (on-harmony): <=6 labels, evenly spaced over volume rank, >=2000 vox
            vol = {l: int((lbl == l).sum()) for l in labs}
            big = sorted([l for l in labs if vol[l] >= 2000], key=lambda l: -vol[l])
            labs = [big[i] for i in np.unique(np.linspace(0, len(big) - 1, MAX_LABELS).round().astype(int))]
        for lab in labs:
            M = lbl == lab
            if M.sum() < 20:
                continue
            idx = np.argwhere(M)
            pad = np.ceil(PAD_MM / zooms).astype(int)
            lo = np.maximum(idx.min(0) - pad, 0)
            hi = np.minimum(idx.max(0) + pad + 1, M.shape)
            cs = tuple(slice(a_, b_) for a_, b_ in zip(lo, hi))

            def rs(a, interp):
                s = sitk.GetImageFromArray(np.ascontiguousarray(a.transpose(2, 1, 0)))
                s.SetSpacing(tuple(float(x) for x in zooms))
                size = [max(int(round(n * zz)), 1) for n, zz in zip(a.shape, zooms)]
                out = sitk.Resample(s, size, sitk.Transform(), interp, s.GetOrigin(), (1.0, 1.0, 1.0), s.GetDirection(), 0.0, s.GetPixelID())
                return sitk.GetArrayFromImage(out).transpose(2, 1, 0)
            z1 = rs(z[cs].astype(np.float32), sitk.sitkLinear)
            m1 = rs(M[cs].astype(np.uint8), sitk.sitkNearestNeighbor) > 0
            f1 = rs(fg[cs].astype(np.uint8), sitk.sitkNearestNeighbor) > 0
            if m1.sum() < MIN_VOX:
                continue
            f = region_row(tex, shp, z1, m1, f1)
            if f is None:
                continue
            rows.append({"key": key, "case": cid, "label": lab, "native_spacing": "x".join(f"{x:.2f}" for x in zooms), **f})
    except Exception as e:  # noqa: BLE001
        print(f"SKIP {key} {cid}: {e!r}", flush=True)
    return (key, cid), rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["brats", "xds", "openms"])
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--nproc", type=int, default=4)
    p.add_argument("--maxc", type=int, default=20)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--tag", default="")
    p.add_argument("--only-keys", default=None, help="comma list (testing)")
    a = p.parse_args()
    shard = DATA / f"radiomics_{a.task}{a.tag}_shard{a.rank}.csv"
    if a.task == "brats":
        units = [(p_, ) for p_ in sorted(pd.read_csv(DATA / "patient_region_deltas.csv")["case"].unique())]  # all 71 (no subsampling)
        fn, ukey = brats_unit, lambda u: u[0]
        arg = lambda u: u[0]  # noqa: E731
        doneid = lambda r: r["patient"]  # noqa: E731
    else:
        if a.task == "openms":
            assert (DATA / "radiomics_xds_frozen.json").exists(), "frozen model must exist before open-ms extraction"
            keys = sorted(k for k in xd.MAN if k.startswith("open-ms"))
        else:
            cells = xd.dev_cells()
            keys = sorted({k for k in set(cells.train) | set(cells["eval"]) if not k.startswith("brats:")})
        if a.only_keys:
            keys = [k for k in keys if k in a.only_keys.split(",")]
        units = []
        for k in keys:
            mand, fn_ = xd.MAN[k]
            units += [(k, c, i, l, mand) for c, i, l in fn_()[: a.maxc]]
        fn, ukey, arg = xds_unit, (lambda u: (u[0], u[1])), (lambda u: u)
        doneid = lambda r: (r["key"], r["case"])  # noqa: E731
    if a.limit:
        units = units[: a.limit]
    mine = units[a.rank::a.world_size]
    done = set()
    if shard.exists():
        d = pd.read_csv(shard, usecols=["patient"] if a.task == "brats" else ["key", "case"])
        done = set(d["patient"]) if a.task == "brats" else set(zip(d["key"], d["case"].astype(str)))
    todo = [u for u in mine if (ukey(u) if a.task == "brats" else (ukey(u)[0], str(ukey(u)[1]))) not in done]
    print(f"{a.task} rank {a.rank}/{a.world_size}: {len(mine)} units, {len(todo)} todo", flush=True)
    header = not shard.exists()
    t0 = time.time()
    with Pool(a.nproc, initializer=_init) as pool:
        for i, (uid, rows) in enumerate(pool.imap_unordered(fn, [arg(u) for u in todo], chunksize=1)):
            if rows:
                pd.DataFrame(rows).to_csv(shard, mode="a", header=header, index=False)
                header = False
            else:  # record empty units so resume skips them (marker row of NaNs would break columns; log only)
                print("EMPTY", uid, flush=True)
            print(f"{i+1}/{len(todo)} {uid} rows={len(rows)} t={time.time()-t0:.0f}s", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
