#!/usr/bin/env python
"""
xds_full_hand.py -- generic (non-BraTS) version of bigfeat_extract.py: the same ~105 hand-made features
(first-order, texture/acf/GLCM/LBP, R-vs-ring contrast, boundary, shape; for target region R (eroded 1 vox), RING
(1-5 vox outside R, inside foreground) and WHOLE foreground), same function bodies (imported from bigfeat_extract).
Differences from the BraTS version (stated, not hidden): the "brain" mask is the intensity foreground (>2 % of the
p99.5 range, plus the target); the target region is the POOLED foreground label (all labels > 0, or label 1 for
mandible-only sets) exactly as the ladder pools labels; images are resampled to 1 mm isotropic first (same as
radiomics_extract.xds_unit) so voxel-scale features are comparable with BraTS (1 mm).
Features only -- no outcomes are read.  Sharded + resumable:
  python xds_full_hand.py --rank R --world-size W [--nproc 4] [--maxc 20] [--only-keys k1,k2]
Output: outputs/data/xds_full_hand_shard{R}.csv  (key, case, n_R, n_ring, R_*, ring_*, brain_*, con_*, bnd_*, shape_*)
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
import xds_full_common as C  # noqa: E402
import bigfeat_extract as bf  # noqa: E402
import nibabel as nib  # noqa: E402
from scipy.ndimage import distance_transform_edt, gaussian_filter, laplace  # noqa: E402
from scipy.stats import ks_2samp, rankdata  # noqa: E402


def _rs(a, zooms, interp):
    import SimpleITK as sitk
    s = sitk.GetImageFromArray(np.ascontiguousarray(a.transpose(2, 1, 0)))
    s.SetSpacing(tuple(float(x) for x in zooms))
    size = [max(int(round(n * zz)), 1) for n, zz in zip(a.shape, zooms)]
    out = sitk.Resample(s, size, sitk.Transform(), interp, s.GetOrigin(), (1.0, 1.0, 1.0), s.GetDirection(), 0.0, s.GetPixelID())
    return sitk.GetArrayFromImage(out).transpose(2, 1, 0)


def unit(task):
    import SimpleITK as sitk
    key, cid, ip, lp, mand = task
    try:
        im = nib.load(str(ip))
        img = np.squeeze(np.asarray(im.dataobj)).astype(np.float32)
        while img.ndim > 3:
            img = img[..., 0]
        zooms = np.asarray(im.header.get_zooms()[:3], float)
        lbl = np.squeeze(np.asarray(nib.load(str(lp)).dataobj)).round().astype(int)
        mask = (lbl == 1) if mand else (lbl > 0)
        lo0, hi0 = float(img.min()), float(np.percentile(img, 99.5))
        fg = (img > lo0 + 0.02 * (hi0 - lo0)) | mask
        x = _rs(img, zooms, sitk.sitkLinear)
        mask = _rs(mask.astype(np.uint8), zooms, sitk.sitkNearestNeighbor) > 0
        brain = (_rs(fg.astype(np.uint8), zooms, sitk.sitkNearestNeighbor) > 0) | mask
        del img, fg
        if mask.sum() < bf.MIN_VOX or brain.sum() < 1000:
            return (key, cid), []
        v = x[brain]
        z = np.where(brain, (x - v.mean()) / max(v.std(), 1e-6), 0).astype(np.float32)
        b = brain.astype(np.float32)
        smooth = gaussian_filter(z * b, 2.0) / np.maximum(gaussian_filter(b, 2.0), 1e-6)
        hp = np.where(brain, z - smooth, 0).astype(np.float32)
        g = np.gradient(z)
        gm = np.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2).astype(np.float32)
        del g
        lap = laplace(z).astype(np.float32)
        lo, hi = np.percentile(z[brain], [1, 99])
        gm_brain = float(gm[brain].mean())
        f = {"key": key, "case": cid}
        bidx = np.argwhere(brain)
        bcent = bidx.mean(0)
        brain_f = {f"brain_{k}": vv for k, vv in bf.part_feats(z, hp, gm, lap, brain, lo, hi).items()}
        R = mask
        n_full = int(R.sum())
        idx = np.argwhere(R)
        lo_i = np.maximum(idx.min(0) - bf.MARGIN, 0)
        hi_i = np.minimum(idx.max(0) + bf.MARGIN + 1, R.shape)
        cs = tuple(slice(a, c) for a, c in zip(lo_i, hi_i))
        Rc, brc = R[cs], brain[cs]
        edt_in = distance_transform_edt(Rc)
        Re = edt_in > 1.0
        if int(Re.sum()) < bf.MIN_VOX:
            Re = Rc.copy()
        shell = Rc & ~(edt_in > 1.0)
        dout = distance_transform_edt(~Rc)
        ring = (dout > 1.0) & (dout <= bf.RING_D) & brc & ~Rc
        cen = np.argwhere(Rc).mean(0) + lo_i
        f.update({"n_R": int(Re.sum()), "n_ring": int(ring.sum())})
        zc, hpc, gmc, lapc = z[cs], hp[cs], gm[cs], lap[cs]
        f.update({f"R_{k}": vv for k, vv in bf.part_feats(zc, hpc, gmc, lapc, Re, lo, hi).items()})
        f.update({f"ring_{k}": vv for k, vv in bf.part_feats(zc, hpc, gmc, lapc, ring, lo, hi).items()})
        f.update(brain_f)
        a_, b_ = zc[Re], zc[ring]
        if a_.size >= bf.MIN_VOX and b_.size >= bf.MIN_VOX:
            pooled = np.sqrt((a_.var() + b_.var()) / 2)
            f["con_d"] = float((a_.mean() - b_.mean()) / max(pooled, 1e-6))
            a2, b2 = bf.sub(a_, 20000), bf.sub(b_, 20000)
            r = rankdata(np.concatenate([a2, b2]))
            auc = (r[: len(a2)].sum() - len(a2) * (len(a2) + 1) / 2) / (len(a2) * len(b2))
            f["con_auc"] = float(max(auc, 1 - auc))
            f["con_ks"] = float(ks_2samp(a2, b2).statistic)
        else:
            f.update(con_d=np.nan, con_auc=np.nan, con_ks=np.nan)
        f["bnd_grad_ratio"] = float(gmc[shell].mean() / gm_brain) if shell.sum() >= bf.MIN_VOX else np.nan
        f.update(shape_logvol=float(np.log10(n_full)), shape_surf_vol=float(shell.sum() / n_full),
                 shape_cent_dist=float(np.linalg.norm(cen - bcent)),
                 shape_ring_other_tumor=0.0)
        return (key, cid), [f]
    except Exception as e:  # noqa: BLE001
        print(f"SKIP {key} {cid}: {e!r}", flush=True)
        return (key, cid), []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--nproc", type=int, default=4)
    p.add_argument("--maxc", type=int, default=C.CAP)
    p.add_argument("--only-keys", default=None)
    p.add_argument("--tag", default="")
    a = p.parse_args()
    MAN = C.manifest()
    keys = sorted(set(C.dev_feature_keys()) | set(C.TEST_KEYS))
    if a.only_keys:
        keys = [k for k in keys if k in a.only_keys.split(",")]
    units = []
    for k in keys:
        mand, fn = MAN[k]
        units += [(k, c, i, l, mand) for c, i, l in fn()[: a.maxc]]
    mine = units[a.rank::a.world_size]
    shard = C.DATA / f"xds_full_hand{a.tag}_shard{a.rank}.csv"
    done = set()
    if shard.exists():
        d = pd.read_csv(shard, usecols=["key", "case"])
        done = set(zip(d["key"], d["case"].astype(str)))
    todo = [u for u in mine if (u[0], str(u[1])) not in done]
    print(f"hand rank {a.rank}/{a.world_size}: {len(mine)} units, {len(todo)} todo", flush=True)
    header = not shard.exists()
    t0 = time.time()
    with Pool(a.nproc) as pool:
        for i, (uid, rows) in enumerate(pool.imap_unordered(unit, todo, chunksize=1)):
            if rows:
                pd.DataFrame(rows).to_csv(shard, mode="a", header=header, index=False)
                header = False
            else:
                print("EMPTY", uid, flush=True)
            print(f"{i+1}/{len(todo)} {uid} t={time.time()-t0:.0f}s", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
