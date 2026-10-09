#!/usr/bin/env python
"""
"Train me" (interactive slideshow): let a human go through what the rung-4 (noise-fill) and rung-5 (real-fill) models were trained on, then take
their test. Builds ONE self-contained HTML slideshow (base64 PNGs, no server; arrows = navigate, space = toggle labels, p = toggle the\nmodels' predictions on test slides) in four phases: noise-fill training set -> test set A -> real-fill training set (other\ncases) -> test set B (other cases). Parts:
  A. training set of the NOISE-FILL model: N samples = fold-0 training cases, nnU-Net z-scored (mask norm), patch of the
     plans' patch size around the tumour, passed through the REAL AugLab GPU pipeline of rung 4
     (transform_params_gpu_baseline_kmeans_label_remap_voronoi_spatialDA_train050.json), one seed per sample;
     axial slice at the largest edema cross-section. Labels toggled by a button (the model saw them at every step).
  B. the same cases / seeds through the rung-5 pipeline (transform_params_gpu_v26_6_2_synth_spatialDA_train050.json).
     Both configs apply the synthesis with p = 0.5, so about half the samples are plain T2w: that is what the models saw.
  C. the test: held-out cases on T1n / T1c / FLAIR (and T2w), image only; "reveal" shows the GT, the noise-fill and the
     real-fill model's actual predictions (fold 0) and their slice Dice for edema.
Usage (GPU job):  .venv/bin/python trainme_build.py --train t2w --n-train 12 --n-test 6
Output: outputs/trainme/trainme_<train>.html (+ the PNGs next to it).
"""
from __future__ import annotations
import argparse, base64, io, json, sys
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent
REPO = THIS.parents[6]
sys.path.insert(0, str(REPO))
from auglab.transforms.gpu.transforms import AugTransformsGPU  # noqa: E402

DS = {"t2w": "Dataset052_BraTS2024GliomaT2w", "t1n": "Dataset051_BraTS2024GliomaT1n", "t2f": "Dataset053_BraTS2024GliomaT2f", "t1c": "Dataset054_BraTS2024GliomaT1c"}
RAW = REPO / "benchmark/02_tasks/brain_tumor/brats2024-glioma/2_nnUNet_brats2024-glioma/raw"
PRE = REPO / "benchmark/02_tasks/brain_tumor/brats2024-glioma/2_nnUNet_brats2024-glioma/preprocessed"
PRED = REPO / "benchmark/02_tasks/brain_tumor/brats2024-glioma/8_results_brats2024-glioma/01_predictions/brats2024_glioma_model"
CFG = REPO / "sub-workspaces/auglab_workspace/AugLab/auglab/configs"
ARMS = {"noise-fill (rung 4)": "transform_params_gpu_baseline_kmeans_label_remap_voronoi_spatialDA_train050.json",
        "real-fill (rung 5)": "transform_params_gpu_v26_6_2_synth_spatialDA_train050.json"}
RUNS = {"t2w": {"noise": "auglab/brats2024-glioma_t2w_baseline_kmeans_label_remap_voronoi_20260805_020659",
                "real": "auglab/brats2024-glioma_t2w_v26_6_2_train050_val000_20261005_222300"}}
LAB = {1: ("NCR", "#4a3aa7"), 2: ("edema", "#eb6834"), 3: ("ET", "#1baf7a"), 4: ("RC", "#eda100")}
OUT = THIS.parent / "outputs" / "trainme"


def load(ds, case, sub="imagesTr"):
    img = np.asarray(nib.load(str(RAW / ds / sub / f"{case}_0000.nii.gz")).dataobj).astype(np.float32)
    return img


def zscore(img):
    fg = img > 0
    z = np.zeros_like(img); z[fg] = (img[fg] - img[fg].mean()) / max(img[fg].std(), 1e-6)
    return z


def patch_box(seg, size):
    idx = np.argwhere(seg > 0); c = idx.mean(0).astype(int); size = np.minimum(size, seg.shape)   # a patch larger than the volume (pelvic MRI) -> whole axis
    lo = np.clip(c - np.array(size) // 2, 0, np.array(seg.shape) - np.array(size)); hi = lo + np.array(size)
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def zoom_box(mask2d, rng, win=96):
    """Window of win x win that CONTAINS the ROI's bounding box (or overlaps it when the ROI is larger), placed at a
    uniformly random offset so the ROI is not systematically centred."""
    H, W = mask2d.shape
    win = min(win, H, W)   # small (cropped) volumes: never ask for a window larger than the slice
    idx = np.argwhere(mask2d)
    if idx.size == 0:
        r0, c0 = rng.integers(0, H - win + 1), rng.integers(0, W - win + 1)
        return slice(r0, r0 + win), slice(c0, c0 + win)
    out = []
    for lo, hi, n in ((idx[:, 0].min(), idx[:, 0].max() + 1, H), (idx[:, 1].min(), idx[:, 1].max() + 1, W)):
        a, b = max(0, hi - win), min(lo, n - win)          # start range that keeps the whole ROI inside
        if a > b:                                           # ROI larger than the window: any start overlapping it
            a, b = max(0, lo - win // 2), min(hi - win // 2, n - win)
        s = int(rng.integers(min(a, b), max(a, b) + 1)); out.append(slice(s, s + win))
    return tuple(out)


def to_png(gray, seg=None, alpha=0.45, outline=False, dpi=110, scale=None):
    scale = scale or (3.4 if max(gray.shape) <= 128 else 1.2)   # zoom windows at ~3.4 px/voxel, full-slice thumbnails small
    fig = plt.figure(figsize=(gray.shape[1] / dpi * scale, gray.shape[0] / dpi * scale), dpi=dpi); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    fg = gray != 0; lo, hi = (np.percentile(gray[fg], [1, 99]) if fg.any() else (0, 1))
    ax.imshow(gray.T, cmap="gray", vmin=lo, vmax=hi, origin="lower", interpolation="nearest")
    if seg is not None:
        for k, (name, col) in LAB.items():
            m = (seg == k).T
            if m.any():
                if outline:
                    ax.contour(m.astype(float), levels=[0.5], colors=[col], linewidths=1.2)
                else:
                    rgba = np.zeros(m.shape + (4,)); rgba[m] = matplotlib.colors.to_rgba(col, alpha); ax.imshow(rgba, origin="lower", interpolation="nearest")
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=dpi); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def dice(a, b):
    s = a.sum() + b.sum(); return float(2 * (a & b).sum() / s) if s else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="t2w"); ap.add_argument("--n-train", type=int, default=12); ap.add_argument("--n-test", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    ds = DS[a.train]; dev = torch.device(a.device)
    plans = json.load(open(PRE / ds / "nnUNetPlans.json")); size = plans["configurations"]["3d_fullres"]["patch_size"]
    splits = json.load(open(PRE / ds / "splits_final.json"))[0]["train"]
    rng = np.random.default_rng(a.seed)
    # ---- A/B: training samples
    pipes = {k: AugTransformsGPU(json_path=str(CFG / v), num_labels=5).to(dev).eval() for k, v in ARMS.items()}
    cards = {k: [] for k in ARMS}
    picked = []
    for case in rng.permutation(splits):
        seg = np.asarray(nib.load(str(RAW / ds / "labelsTr" / f"{case}.nii.gz")).dataobj).astype(np.int16)
        if (seg == 2).sum() < 3000:
            continue
        picked.append(case)
        if len(picked) == 2 * a.n_train:
            break
    arm_of = {case: list(ARMS)[0] if i < a.n_train else list(ARMS)[1] for i, case in enumerate(picked)}  # disjoint cases per arm
    for i, case in enumerate(picked):
        img = zscore(load(ds, case)); seg = np.asarray(nib.load(str(RAW / ds / "labelsTr" / f"{case}.nii.gz")).dataobj).astype(np.int16)
        box = patch_box(seg, size); x = torch.from_numpy(img[box])[None, None].to(dev); y = torch.from_numpy(seg[box].astype(np.float32))[None, None].to(dev)
        z = int(np.argmax((seg[box] == 2).sum((0, 1))))
        for k, pipe in pipes.items():
            if k != arm_of[case]:
                continue
            torch.manual_seed(1000 * a.seed + i); np.random.seed(1000 * a.seed + i)
            with torch.no_grad():
                xo, yo = pipe(x.clone(), y.clone())
            g = xo[0, 0].float().cpu().numpy()[:, :, z]; s = yo[0, 0].round().long().cpu().numpy()[:, :, z]
            changed = float(np.abs(xo[0, 0].float().cpu().numpy() - img[box]).mean()) > 1e-3
            zb = zoom_box(s == 2, rng)
            cards[k].append(dict(case=case, i=i, img=to_png(g[zb]), lab=to_png(g[zb], s[zb]), thumb=to_png(g), augmented=changed))
        print(f"train sample {i + 1}/{len(picked)} {case}", flush=True)
    # ---- C: test
    d = pd.read_csv(THIS.parent / "outputs/data/patient_region_deltas.csv")
    d = d[(d.train == a.train) & (d.region == "SNFH") & (d["eval"] == "t1n")].sort_values("delta_dice")
    worst, best = list(d.case.head(a.n_test)), list(d.case.tail(a.n_test))[::-1]
    tests = [c for pair in zip(worst, best) for c in pair]            # worst1 best1 worst2 best2 ...
    test_set = {c: ("A" if i < a.n_test else "B") for i, c in enumerate(tests)}  # set A after the noise arm, set B after the real arm
    quiz = []
    for case in tests:
        seg = np.asarray(nib.load(str(RAW / ds / "labelsTr" / f"{case}.nii.gz")).dataobj).astype(np.int16) if (RAW / ds / "labelsTr" / f"{case}.nii.gz").exists() else None
        if seg is None:
            gt_path = next((RAW / DS[a.train]).glob(f"labelsTs*/{case}.nii.gz"), None) or next(RAW.glob(f"*/labelsTs*/{case}.nii.gz"))
            seg = np.asarray(nib.load(str(gt_path)).dataobj).astype(np.int16)
        z = int(np.argmax((seg == 2).sum((0, 1))))
        row = dict(case=case, set=test_set[case], views=[]); zb = zoom_box(seg[:, :, z] == 2, rng); sz = seg[:, :, z]
        for ev in ("t1n", "t1c", "t2f", "t2w"):
            img = np.asarray(nib.load(str(RAW / ds / f"imagesTs_{ev}" / f"{case}_0000.nii.gz")).dataobj).astype(np.float32)[:, :, z]
            preds = {}
            for arm, run in RUNS[a.train].items():
                p = np.asarray(nib.load(str(PRED / a.train / run / "fold0" / ev / f"{case}.nii.gz")).dataobj).astype(np.int16)[:, :, z]
                preds[arm] = dict(png=to_png(img[zb], p[zb]), dice=dice(p == 2, sz == 2))
            row["views"].append(dict(ev=ev, img=to_png(img[zb]), thumb=to_png(img), gt=to_png(img[zb], sz[zb]), noise=preds["noise"], real=preds["real"],
                                     dcase=float(d[d.case == case].delta_dice.iloc[0]) if ev == "t1n" else None))
        quiz.append(row); print("test", case, flush=True)
    # ---- HTML slideshow: phases = noise train -> test A -> real train -> test B; keys: arrows, space = labels, p = predictions
    slides = []
    for k in ARMS:
        cards_k = sorted(cards[k], key=lambda c: c["i"])
        set_id = "A" if k == list(ARMS)[0] else "B"
        for j in range(0, len(cards_k), 2):   # training images two by two
            pair = cards_k[j:j + 2]
            slides.append(dict(phase=f"training set, {k} model", n=f"{j + 1}-{j + len(pair)}/{len(cards_k)}", preds=None,
                               items=[dict(title=f"{c['case']}  {'synthesised' if c['augmented'] else 'plain training image'}", img=c["img"], lab=c["lab"], thumb=c["thumb"]) for c in pair]))
        for q in [q for q in quiz if q["set"] == set_id]:
            for v in q["views"]:
                slides.append(dict(phase=f"test set {set_id} (after the {k} model)", n=f"case {q['case']}",
                                   items=[dict(title=f"{v['ev'].upper()}{' (training contrast)' if v['ev'] == a.train else ''}", img=v["img"], lab=v["gt"], thumb=v["thumb"])],
                                   preds=dict(noise=v["noise"]["png"], real=v["real"]["png"], dn=v["noise"]["dice"], dr=v["real"]["dice"], dcase=v["dcase"])))
    H = ["<!doctype html><html><head><meta charset='utf-8'><title>Train me: BraTS fill swap</title><style>",
         "body{font-family:sans-serif;background:#1a1a19;color:#eee;margin:0} #top{padding:10px 16px;font-size:14px;color:#c3c2b7} #main{display:flex;gap:24px;padding:0 16px;align-items:flex-start;flex-wrap:wrap}",
         ".item{display:flex;gap:10px;align-items:flex-start} .item .big{width:560px} .item .th{width:180px} .cap{font-size:12px;color:#c3c2b7;margin-top:4px} .hid{display:none}",
         "#preds{display:flex;gap:8px;padding:8px 16px} #preds img{width:330px} kbd{background:#333;padding:1px 5px;border-radius:3px}",
         "</style></head><body><div id='top'></div><div id='main'></div><div id='preds' class='hid'></div>",
         "<script>const S=", json.dumps(slides), ";let i=0,lab=false,pr=false;const b64=s=>'data:image/png;base64,'+s;",
         "function show(){const s=S[i];document.getElementById('top').innerHTML=`<b>${s.phase}</b> &nbsp; ${s.n} &nbsp; slide ${i+1}/${S.length} &nbsp;&nbsp; <kbd>&larr;</kbd><kbd>&rarr;</kbd> navigate &nbsp; <kbd>space</kbd> labels ${lab?'ON':'off'}`+(s.preds?` &nbsp; <kbd>p</kbd> predictions ${pr?'ON':'off'}`:'');",
         "document.getElementById('main').innerHTML=s.items.map(it=>`<div class='item'><div><img class='big' src='${b64(lab?it.lab:it.img)}'><div class='cap'>${it.title}</div></div><img class='th' src='${b64(it.thumb)}'></div>`).join('');",
         "const P=document.getElementById('preds');if(s.preds&&pr){P.classList.remove('hid');P.innerHTML=`<div><div class='cap'>noise-fill model, edema Dice ${s.preds.dn.toFixed(2)}</div><img src='${b64(s.preds.noise)}'></div><div><div class='cap'>real-fill model, edema Dice ${s.preds.dr.toFixed(2)}${s.preds.dcase!==null?' (patient &Delta; '+(100*s.preds.dcase).toFixed(0)+' pts)':''}</div><img src='${b64(s.preds.real)}'></div>`;}else{P.classList.add('hid');}}",
         "document.addEventListener('keydown',e=>{if(e.key==='ArrowRight'){i=Math.min(i+1,S.length-1);lab=false;pr=false;}else if(e.key==='ArrowLeft'){i=Math.max(i-1,0);lab=false;pr=false;}else if(e.key===' '){e.preventDefault();lab=!lab;}else if(e.key==='p'){pr=!pr;}else return;show();});show();</script></body></html>"]
    out = OUT / f"trainme_{a.train}.html"; out.write_text("".join(H)); print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
