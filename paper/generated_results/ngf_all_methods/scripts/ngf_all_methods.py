#!/usr/bin/env python
"""
NGF texture-preservation for EVERY augmentation method on EVERY ablation task (paper: "texture kept
(NGF)" vs "Dice gain").

For one (task, dataset, contrast) this script:
  1. picks a fixed random subset of N training scans (nnU-Net raw imagesTr/labelsTr, or the BIDS tree
     for ToothFairy2), seed fixed,
  2. for each method builds the AugLab GPU transform from the SAME JSON the trainings used (spatial
     transforms zeroed so the output stays voxel-aligned with the source; optional "noblur" variant),
  3. draws K augmented copies per scan IN MEMORY (whole-image z-score in, like every existing
     generate_*_volumes.py) and scores NGF(source, augmented) with the project's existing
     compute_ngf_texture.ngf_scores (Haber & Modersitzki squared-cosine of gradient directions,
     3-D Sobel) -- NOT re-implemented here,
  4. adds two reference rows per scan: `identity` (augmented == source, NGF 1) and `noise`
     (iid uniform noise, the isotropic floor ~1/3).

ROI (identical to the Open-MS paper table): foreground = source > 0.10 * p99(source) (MR); for CT
`ct` mode foreground = HU > -500; eroded x3 (3x3x3 min-pool). Secondary ROI `lab` = (label > 0)
eroded x1 (skipped if < 50 voxels). NGF is reported as mean over ROI voxels (`ngf_all`) and over the
top-50% source-gradient voxels of the ROI (`ngf_edge`), as in compute_ngf_texture.py.

Usage:  python ngf_all_methods.py --task glioma --dataset brats2024-glioma --contrast t1n ...
        python ngf_all_methods.py --sanity-configs      # builds + prints configs, no GPU
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ngf_all")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
AUGLAB = REPO / "sub-workspaces/auglab_workspace/AugLab"
CFGDIR = AUGLAB / "auglab/configs"
TASKS = REPO / "benchmark/02_tasks"
OMS_TEX = TASKS / "brain_ms/open-ms/7_analysis_open-ms/texture_analysis_lvl_1/scripts"

# ───────────────────────────── dataset registry ─────────────────────────────
# (task_dir, dataset_dir, nnunet Dataset folder or None, mode)
# mode: "mr" -> fg = src > 0.1*p99 ; "ct" -> fg = HU > -500
REG = {
    # key: (task, dataset, contrast, nnunet_dataset_dir, mode)
    "brats2024-glioma/t1n": ("glioma", "brain_tumor/brats2024-glioma", "t1n", "Dataset051_BraTS2024GliomaT1n", "mr"),
    "brats2024-glioma/t2w": ("glioma", "brain_tumor/brats2024-glioma", "t2w", "Dataset052_BraTS2024GliomaT2w", "mr"),
    "brats2024-glioma/t2f": ("glioma", "brain_tumor/brats2024-glioma", "t2f", "Dataset053_BraTS2024GliomaT2f", "mr"),
    "brats2024-glioma/t1c": ("glioma", "brain_tumor/brats2024-glioma", "t1c", "Dataset054_BraTS2024GliomaT1c", "mr"),
    "open-ms/flair": ("ms", "brain_ms/open-ms", "flair", "Dataset070_OpenMS_FLAIR", "mr"),
    "open-ms/t1w": ("ms", "brain_ms/open-ms", "t1w", "Dataset071_OpenMS_T1W", "mr"),
    "ispy2/t1wce": ("breast", "breast_cancer/ispy2", "t1wce", "Dataset100_ISPY2T1wce", "mr"),
    "ispy2/t2w": ("breast", "breast_cancer/ispy2", "t2w", "Dataset101_ISPY2T2w", "mr"),
    "chaos/t1in": ("abdomen", "abdomen_healthy/chaos", "t1in", "Dataset060_CHAOS_MR_T1in", "mr"),
    "chaos/t2spir": ("abdomen", "abdomen_healthy/chaos", "t2spir", "Dataset061_CHAOS_MR_T2spir", "mr"),
    "on-harmony/t1w": ("brain_healthy", "brain_healthy/on-harmony", "t1w", "Dataset031_OnHarmonyT1w31", "mr"),
    "on-harmony/t2w": ("brain_healthy", "brain_healthy/on-harmony", "t2w", "Dataset032_OnHarmonyT2w31", "mr"),
    "on-harmony/dwi": ("brain_healthy", "brain_healthy/on-harmony", "dwi", "Dataset033_OnHarmonyDWI31", "mr"),
    "toothfairy2/cbct": ("mandible", "mandible_healthy/toothfairy2", "cbct", None, "ct"),
    "totalseg-pelvic/ct": ("pelvis", "pelvis_healthy/totalseg-pelvic", "ct", "Dataset130_TotalsegPelvic_CT", "ct"),
    "totalseg-pelvic/mri": ("pelvis", "pelvis_healthy/totalseg-pelvic", "mri", "Dataset131_TotalsegPelvic_MRI", "mr"),
}


def list_cases(key):
    task, ddir, contrast, nn, mode = REG[key]
    root = TASKS / ddir
    items = []
    if nn is not None:
        base = next(root.glob("2_nnUNet_*/raw")) / nn
        for img in sorted((base / "imagesTr").glob("*_0000.nii.gz")):
            cid = img.name[: -len("_0000.nii.gz")]
            lab = base / "labelsTr" / f"{cid}.nii.gz"
            if lab.exists():
                items.append((cid, img, lab))
        n_labels = len(json.load(open(base / "dataset.json"))["labels"])
    else:  # ToothFairy2: BIDS (already 0.6 mm iso RAS, 3 reduced labels)
        bids = root / "1_BIDS_toothfairy2/maxillofacial-toothfairy2"
        for sd in sorted(bids.glob("sub-*")):
            if not sd.is_dir():
                continue
            s = sd.name
            img = sd / "anat" / f"{s}_acq-cbct_ct.nii.gz"
            lab = bids / "derivatives/labels" / s / "anat" / f"{s}_acq-cbct_ct_label-maxillofacial_seg.nii.gz"
            if img.exists() and lab.exists():
                items.append((s, img, lab))
        n_labels = 4
    return items, n_labels


# ───────────────────────────── config building ─────────────────────────────
SPATIAL = ("FlipTransform", "AffineTransform", "nnUNetSpatialTransform")

# method key -> (source json, mode)   mode "only:<Key>" keep only that block active
METHODS = {
    "palette":        ("transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json", None),
    # retired rung 4 (one noise level per label); building it now needs AUGLAB_ALLOW_LEGACY_NOISEFILL=1
    "noisefill":      ("transform_params_gpu_VALsynthonly_ImageContrastV26_6_2NoiseFillV2GPUTransform.json", None),
    # current rung 4 (2026-10-07, label_voronoi: Voronoi cells inside noise-refilled labels)
    "noisefill_lblvor": ("transform_params_gpu_VALsynthonly_ImageContrastV26_6_2NoiseFillV2LblVorGPUTransform.json", None),
    # REAL USE (2026-10-08, Paul): the ladder rungs measured with their exact TRAINING configs, i.e. the synthesis applied with
    # its training probability (0.5; the other draws are the source with spatial transforms zeroed, NGF 1), like every
    # competitor and PALETTE-Aug above. The VALsynthonly keys force the synthesis on every draw (probability 1).
    "palette_train050":          ("transform_params_gpu_v26_6_2_synth_spatialDA_train050.json", None),
    "noisefill_lblvor_train050": ("transform_params_gpu_baseline_kmeans_label_remap_voronoi_lblvor_spatialDA_train050.json", None),
    "kmeans_remap_train050":     ("transform_params_gpu_baseline_kmeans_label_remap_spatialDA_train050.json", None),
    "kmeans_train050":           ("transform_params_gpu_baseline_kmeans_spatialDA_train050.json", None),
    "kmeans":         ("transform_params_gpu_VALsynthonly_kmeans.json", None),
    "kmeans_remap":   ("transform_params_gpu_VALsynthonly_kmeans_label_remap.json", None),
    "auglab_default": ("transform_params_gpu_default01-23.json", None),
    "synthseg_noEM":  ("transform_params_gpu_default01-23_Synthseg.json", "only:SynthSeg"),
    "synthseg_EM":    ("transform_params_gpu_default01-23_Synthseg_EM.json", "only:SynthSeg"),
    "srcsm":          ("transform_params_gpu_srcsm_semrandconv.json", None),
    "ours_train050":  ("transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json", None),
}


def build_cfg(method, variant):
    src, mode = METHODS[method]
    cfg = json.load(open(CFGDIR / src))
    cfg = copy.deepcopy(cfg)
    if mode and mode.startswith("only:"):
        keep = mode.split(":")[1]
        for k, v in cfg.items():
            if isinstance(v, dict) and "probability" in v and k != keep:
                v["probability"] = 0.0
    for k in SPATIAL:
        if k in cfg and isinstance(cfg[k], dict) and "probability" in cfg[k]:
            cfg[k]["probability"] = 0.0
    if "nnUNetSpatialTransform" in cfg:   # inert in AugTransformsGPU, zero for clarity
        cfg["nnUNetSpatialTransform"]["p_elastic_deform"] = 0.0
    if variant == "noblur":
        for k in ("ImageContrastV26_6_2GPUTransform", "ImageContrastV26_6_2NoiseFillGPUTransform"):
            if k in cfg:
                cfg[k]["blur_sigmas"] = [0.0]
        if "SynthSeg" in cfg and cfg["SynthSeg"].get("probability", 0) > 0:
            s = cfg["SynthSeg"]
            s["randomise_res"] = False; s["blur_range"] = 1.0
            s["data_res"] = 1.0; s["atlas_res"] = 1.0; s["thickness"] = None
        for k in ("GaussianBlurTransform", "SimulateLowResTransform"):
            if k in cfg:
                cfg[k]["probability"] = 0.0
    return cfg


def active(cfg):
    return {k: v["probability"] for k, v in cfg.items() if isinstance(v, dict) and v.get("probability", 0) > 0}


# ───────────────────────────── main ─────────────────────────────
def foreground(src_raw, mode):
    import torch
    if mode == "ct":
        return src_raw > -500.0
    p99 = torch.quantile(src_raw.flatten()[:: max(1, src_raw.numel() // 4_000_000)], 0.99)  # strided subsample (quantile cap)
    return src_raw > 0.10 * p99


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", help="registry key, e.g. brats2024-glioma/t1n")
    ap.add_argument("--n-scans", type=int, default=20)
    ap.add_argument("--n-draws", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", default="noblur", help="comma list of noblur,asblur")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--erode", type=int, default=3)
    ap.add_argument("--max-vox", type=int, default=12_000_000, help="centre-crop scans larger than this (GPU memory)")
    ap.add_argument("--out", type=Path, required=False)
    ap.add_argument("--max-scans-time", type=float, default=0, help="stop after this many seconds (0 = no limit)")
    ap.add_argument("--print-configs", action="store_true")
    args = ap.parse_args()

    methods = [m for m in args.methods.split(",") if m]
    variants = [v for v in args.variants.split(",") if v]
    if args.print_configs:
        for v in variants:
            for m in methods:
                print(v, m, active(build_cfg(m, v)))
        return

    import torch
    if not torch.cuda.is_available():
        sys.exit("CUDA not available on this node -- refusing to run (resubmit excluding the node)")
    sys.path.insert(0, str(AUGLAB)); sys.path.insert(0, str(OMS_TEX))
    import nibabel as nib
    from auglab.transforms.gpu.transforms import AugTransformsGPU
    from compute_ngf_texture import ngf_map, gradient_3d

    def ngf_scores(src, gen, mask, edge_pctl=50):
        # same as compute_ngf_texture.ngf_scores but torch.quantile is capped at 16M elements
        # (large CT/CBCT volumes) -> use kthvalue for the 50th-percentile edge gate.
        sim, g2 = ngf_map(src, gen)
        v_sim, v_g2 = sim[mask], g2[mask]
        n = int(v_sim.numel())
        if n == 0:
            return float("nan"), float("nan"), 0
        k = max(1, int(np.ceil(n * edge_pctl / 100.0)))
        thr = torch.kthvalue(v_g2, k).values
        sel = v_g2 >= thr
        return float(v_sim.mean()), float(v_sim[sel].mean()), n
    from compute_texture_metrics_openms import erode

    dev = torch.device("cuda")
    task, ddir, contrast, nn, mode = REG[args.key]
    dataset = ddir.split("/")[1]
    cases, n_labels = list_cases(args.key)
    rng = np.random.default_rng(args.seed)
    n = min(args.n_scans, len(cases))
    idx = np.sort(rng.choice(len(cases), size=n, replace=False))
    sel = [cases[i] for i in idx]
    log.info("%s: %d cases available, using N=%d (seed %d), draws=%d, variants=%s, methods=%s",
             args.key, len(cases), n, args.seed, args.n_draws, variants, methods)

    tfs = {}
    for v in variants:
        for m in methods:
            cfg = build_cfg(m, v)
            tmp = Path(os.environ.get("SLURM_TMPDIR", "/tmp")) / f"ngfcfg_{v}_{m}_{os.getpid()}.json"
            tmp.write_text(json.dumps(cfg))
            tfs[(v, m)] = AugTransformsGPU(json_path=str(tmp), num_labels=n_labels).to(dev).eval()
            tmp.unlink()
            log.info("built %s/%s active=%s", v, m, active(cfg))

    rows, fails = [], 0
    t0 = time.time()
    mnames = list(METHODS)
    for ci, (cid, ipath, lpath) in enumerate(sel):
        if args.max_scans_time and time.time() - t0 > args.max_scans_time:
            log.warning("time budget hit after %d scans", ci); break
        nii = nib.load(str(ipath))
        arr = np.asarray(nii.get_fdata(dtype=np.float32))
        if arr.ndim == 4:
            arr = arr[..., 0]
        lab = np.round(np.asarray(nib.load(str(lpath)).get_fdata())).astype(np.int64)
        if lab.shape != arr.shape:
            log.warning("%s label/img shape mismatch %s vs %s -- skipped", cid, lab.shape, arr.shape); continue
        crop_info = ""
        if arr.size > args.max_vox:
            fgb = np.argwhere(arr > (-500.0 if mode == "ct" else 0.1 * np.percentile(arr[::7, ::7, ::7], 99)))
            ctr = (fgb.min(0) + fgb.max(0)) // 2 if len(fgb) else np.array(arr.shape) // 2
            thin = [a for a in range(3) if arr.shape[a] <= 64]            # never shrink thin axes
            k = 3 - len(thin)
            sc = (args.max_vox / (arr.size)) ** (1.0 / k) if k else 1.0
            sl = []
            for a in range(3):
                L = arr.shape[a] if a in thin else max(64, int(arr.shape[a] * sc))
                lo = int(np.clip(ctr[a] - L // 2, 0, arr.shape[a] - L))
                sl.append(slice(lo, lo + L))
            arr, lab = arr[tuple(sl)], lab[tuple(sl)]
            crop_info = f" CROPPED to {arr.shape} (> {args.max_vox/1e6:.0f}M vox)"
        raw = torch.from_numpy(np.ascontiguousarray(arr)).to(dev)
        src = (raw - raw.mean()) / raw.std().clamp_min(1e-8)         # whole-image z-score (all generators)
        lab_t = torch.from_numpy(lab).to(dev)
        fg = foreground(raw, mode)
        sgz, sgy, sgx = gradient_3d(src)
        nonflat = (sgz * sgz + sgy * sgy + sgx * sgx) > 1e-8       # drop exactly-flat voxels (padding/saturation)
        del sgz, sgy, sgx
        rois = {"fg": erode(fg, args.erode) & nonflat, "lab": erode(lab_t > 0, 1) & nonflat}
        flat_frac = float(1.0 - (erode(fg, args.erode) & nonflat).sum() / max(1, int(erode(fg, args.erode).sum())))
        img_t, lbl_t = src[None, None], lab_t[None, None]

        def score(gen, method, variant, draw):
            for rname, m in rois.items():  # noqa: F821  (closure; del only after the last score() call)
                if int(m.sum()) < 50:
                    continue
                a, e, nv = ngf_scores(src, gen, m)  # noqa: F821
                rows.append(dict(task=task, dataset=dataset, contrast=contrast, variant=variant, method=method,
                                 scan=cid, draw=draw, roi=rname, n_vox=nv, flat_frac=flat_frac, ngf_all=a, ngf_edge=e))

        g = torch.Generator(device=dev); g.manual_seed(args.seed * 7919 + ci)
        score(src.clone(), "identity", "ref", 0)
        for d in range(args.n_draws):
            g.manual_seed(args.seed * 7919 + ci * 131 + d)
            score(torch.rand(src.shape, device=dev, generator=g), "noise", "ref", d)
        for (v, m), tf in tfs.items():
            for d in range(args.n_draws):
                seed = (1000 + ci * 100000 + (mnames.index(m) + 1) * 1000 + d) & 0x7FFFFFFF
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
                try:
                    with torch.no_grad():
                        out, _ = tf(img_t.clone(), lbl_t.clone())
                    gen = out[0, 0].float()
                    if gen.shape != src.shape or not torch.isfinite(gen).all():
                        raise RuntimeError(f"bad output shape/finite {tuple(gen.shape)}")
                    score(gen, m, v, d)
                except Exception:
                    fails += 1
                    log.warning("FAIL %s %s/%s draw%d: %s", cid, v, m, d, traceback.format_exc().splitlines()[-1])
                    torch.cuda.empty_cache()
        log.info("scan %d/%d %s shape=%s%s flat_frac=%.3f done (%.0fs elapsed, %d fails)", ci + 1, n, cid, tuple(arr.shape), crop_info, flat_frac, time.time() - t0, fails)
        del raw, src, lab_t, img_t, lbl_t, rois, fg
        torch.cuda.empty_cache()

    import pandas as pd
    df = pd.DataFrame(rows)
    out = args.out or (HERE.parent / "data" / f"ngf_{args.key.replace('/', '__')}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("wrote %d rows -> %s (fails=%d)", len(df), out, fails)
    if len(df):
        print(df[df.roi == "fg"].groupby(["variant", "method"])[["ngf_all", "ngf_edge"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
