#!/usr/bin/env python
"""
Runs all 4 trained cross-contrast synthesis models on the 70 fill-swap eval patients (sliding
window inference), and computes, per patient x region x variant x (source,target) ordered pair,
R2 = squared Pearson correlation between predicted and real target content within that region
(both z-scored, per synth_common's cache convention). "variant" is raw or highpass (image minus
a mask-normalized Gaussian(sigma=2) smooth, applied identically to prediction and real target).

R2(train | eval) for a ladder cell (train=X, eval=Y) is exactly the row source=Y, target=X here
(the model that consumes the EVAL contrast's own real image, checked for how well it recovers the
TRAIN contrast's real content) -- see synthesis_vs_fill_swap_summary.py for that lookup.

Also dumps 2 patients' predictions as NIfTI (native 240x240x155 grid, original affine) under
$SCRATCH/brats_synth_qc/<patient>/ for visual QC, and reports mean healthy-region raw R2 per
(source,target) pair as a sanity gate (expect > 0.3 -- if not, the models did not learn and
downstream numbers should not be trusted).

Usage (inside a Slurm GPU job):
  .venv/bin/python infer_synth.py [--n-qc-patients 2] [--smoke]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from scipy.ndimage import gaussian_filter

import synth_common as sc
from train_synth import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROI = (96, 96, 96)
HP_SIGMA = 2.0


def load_model(source: str, device):
    ckpt_dir = sc.CKPT_DIR / source
    path = ckpt_dir / "final.pt"
    if not path.exists():
        path = ckpt_dir / "latest.pt"
    ck = torch.load(path, map_location=device)
    model = build_model().to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, tuple(ck["targets"]), path


def r2_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    r = np.corrcoef(a, b)[0, 1]
    return float(r * r)


def mask_normalized_highpass(vol: np.ndarray, mask: np.ndarray, sigma=HP_SIGMA) -> np.ndarray:
    volf = (vol * mask).astype(np.float32)
    num = gaussian_filter(volf, sigma=sigma, mode="constant")
    den = gaussian_filter(mask.astype(np.float32), sigma=sigma, mode="constant")
    smooth = np.zeros_like(vol, dtype=np.float32)
    valid = den > 1e-3
    smooth[valid] = num[valid] / den[valid]
    hp = vol.astype(np.float32) - smooth
    hp[~mask] = 0.0
    return hp


def predict(model, x: np.ndarray, device) -> np.ndarray:
    t = torch.from_numpy(x).to(device)[None, None]  # (1,1,D,H,W)
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        out = sliding_window_inference(t, roi_size=ROI, sw_batch_size=4, predictor=model,
                                        overlap=0.5, mode="gaussian")
    return out[0].float().cpu().numpy()  # (3,D,H,W)


def uncrop(vol: np.ndarray, cache: dict) -> np.ndarray:
    full = np.zeros(tuple(cache["orig_shape"]), dtype=np.float32)
    lo, hi = cache["bbox_lo"], cache["bbox_hi"]
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    full[sl] = vol
    return full


def patient_affine(pid: str) -> np.ndarray:
    anat = sc.BIDS_ROOT / f"sub-{pid}" / "anat"
    f = anat / f"sub-{pid}_{sc.ngf.CONTRAST_SUFFIX['t1n']}"
    return nib.load(str(f)).affine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-qc-patients", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    split = json.loads((sc.OUT_DIR / "data" / "synth_train_val_split.json").read_text())
    eval_ids = split["eval"]
    if args.smoke:
        eval_ids = eval_ids[: args.n_qc_patients]

    models = {}
    for src in sc.CONTRASTS:
        model, targets, path = load_model(src, device)
        models[src] = (model, targets)
        log.info("loaded %s model from %s (targets=%s)", src, path, targets)

    for pid in eval_ids:
        sc.build_one_cache(pid)  # idempotent, no-op if already cached

    data_dir = sc.OUT_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_csv = data_dir / "synthesis_r2_per_patient.csv"
    qc_dir = sc.QC_DIR
    qc_dir.mkdir(parents=True, exist_ok=True)

    healthy_r2s = {}  # (source,target) -> list of raw R2 for sanity gate
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient", "source", "target", "region", "variant", "n_vox", "r2"])
        for pi, pid in enumerate(eval_ids):
            cache = sc.load_cache(pid)
            for src, (model, targets) in models.items():
                pred = predict(model, cache[src].astype(np.float32), device)  # (3,D,H,W)
                brain = cache["brain"]
                pred = pred * brain[None]

                pred_hp = np.stack([mask_normalized_highpass(pred[ti], brain)
                                     for ti in range(len(targets))], axis=0)

                for ti, tgt in enumerate(targets):
                    true_raw = cache[tgt].astype(np.float32)
                    true_hp = mask_normalized_highpass(true_raw, brain)
                    for region in sc.REGIONS:
                        m = cache[f"mask_{region}"]
                        n = int(m.sum())
                        if n < sc.MIN_REGION_VOX:
                            continue
                        r2_raw = r2_pearson(pred[ti][m], true_raw[m])
                        r2_hp = r2_pearson(pred_hp[ti][m], true_hp[m])
                        writer.writerow([pid, src, tgt, region, "raw", n, r2_raw])
                        writer.writerow([pid, src, tgt, region, "highpass", n, r2_hp])
                        if region == "healthy":
                            healthy_r2s.setdefault((src, tgt), []).append(r2_raw)

                if pi < args.n_qc_patients:
                    out_pid_dir = qc_dir / pid
                    out_pid_dir.mkdir(parents=True, exist_ok=True)
                    aff = patient_affine(pid)
                    for ti, tgt in enumerate(targets):
                        full = uncrop(pred[ti], cache)
                        nib.save(nib.Nifti1Image(full, aff),
                                  str(out_pid_dir / f"pred_{src}_to_{tgt}.nii.gz"))
            f.flush()
            log.info("patient %d/%d (%s) done", pi + 1, len(eval_ids), pid)

    log.info("wrote %s", out_csv)
    log.info("QC NIfTIs under %s (patients: %s)", qc_dir, eval_ids[: args.n_qc_patients])

    log.info("=== SANITY: healthy-brain raw R2 per (source->target) ===")
    all_ok = True
    for (src, tgt), vals in sorted(healthy_r2s.items()):
        m = float(np.nanmean(vals))
        ok = m > 0.3
        all_ok &= ok
        log.info("  %s -> %s : healthy R2 = %.3f  %s", src, tgt, m, "OK" if ok else "*** LOW ***")
    log.info("SANITY GATE: %s", "PASSED (all > 0.3)" if all_ok else "FAILED -- inspect before trusting downstream numbers")


if __name__ == "__main__":
    main()
