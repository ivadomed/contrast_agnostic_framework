#!/usr/bin/env python
"""
Port of the BraTS2024-glioma "does real-fill help or hurt cross-contrast?" texture
analysis (benchmark/02_tasks/brain_tumor/brats2024-glioma/7_analysis_brats2024-glioma/texture_analysis_lvl_1/
FINDINGS.md, esp. sections 1/3/7) to the breast task (ispy2 trains t1wce/t2w; eval on
ispy2's own held-out other contrast + duke-breast-mri t1wce_uni/precontrast_uni).

Single target label ("tumour"). Breast contrasts are NOT co-registered across patients'
series in general, so unlike BraTS this script does ONLY per-image measures (region vs
its own surrounding ring, or region-internal depth structure, computed independently per
image) -- no NGF / correlation-ratio / cross-contrast voxel alignment.

Reused (imported unchanged): stat_tests.wilcoxon_p. step_d/step_auc/ramp-R are a direct
re-implementation of BraTS compute_step_affine.py / internal_ramp_vs_fill_swap.py math
(can't import directly -- wired to BraTS's 4-contrast BIDS loader).

WORKER (this file): sharded (--rank/--world-size), resumable -- appends one row-set per
(pair, case) to this shard's own CSVs, skips cases already done on rerun. Crops every
volume to the GT bounding box (+pad) before any distance transform -- breast tumours are
tiny relative to the full FOV, so this is the dominant speedup over the first (timed-out)
version. Subsamples to <=40 cases per (train, eval_source, eval_item) row (seed 0), per
Doppel's instruction, since most wall-clock is large-image I/O.

Usage (CPU-only Vulcan Slurm job via run_job, never the login node):
  .venv/bin/python breast_texture_analysis.py --rank R --world-size W
Then reduce with breast_texture_summarize.py once all shards finish.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, find_objects
from scipy.stats import mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parents[7]
ISPY2_ROOT = PROJECT_ROOT / "benchmark" / "02_tasks" / "breast_cancer" / "ispy2"
DUKE_ROOT = PROJECT_ROOT / "benchmark" / "02_tasks" / "breast_cancer" / "duke-breast-mri"
OUT = Path(__file__).resolve().parents[1] / "outputs"
DATA_DIR = OUT / "data"
FOLDS = ("fold0", "fold1", "fold2")
LABEL_ID = 1  # tumour
D_OUT = 5     # ring outer radius (voxels), matches BraTS primary config
PAD = D_OUT + 2
MIN_VOX = 50  # breast tumour masks are much smaller than BraTS regions
MAX_CASES_PER_ROW = 40
SEED = 0

ISPY2_DATASET_DIR = {"t1wce": "Dataset100_ISPY2T1wce", "t2w": "Dataset101_ISPY2T2w"}

DELTA_COLS = ["train", "eval_source", "eval_item", "case", "fold", "rung", "in_domain",
              "same_contrast_cross_dataset", "dice", "recall", "precision", "vol_ratio", "fp_share_ring"]
VIS_COLS = ["train", "eval_source", "eval_item", "case", "in_domain",
            "same_contrast_cross_dataset", "step_d", "step_auc", "ramp_R", "gt_vox"]


def load(p: Path) -> np.ndarray:
    return np.asarray(nib.load(str(p)).dataobj).round().astype(np.int16)


def load_img(p: Path) -> np.ndarray:
    return np.asarray(nib.load(str(p)).dataobj).astype(np.float64)


def rung_run_id(run_keys, label, labels) -> str:
    return run_keys[labels.index(label)].split("/")[-1]


def category_for(run_id: str) -> str:
    return "auglab" if "kmeans" in run_id else "nnUNet"


def _ladder(path: Path):
    import json
    return json.loads(path.read_text())


def ispy2_pair(train, eval_item, ladder_json):
    d = _ladder(ladder_json)
    noise_id = rung_run_id(d["run_keys"], "+voronoi (noise fill)", d["labels"])
    real_id = rung_run_id(d["run_keys"], "v26_6_2 (real fill)", d["labels"])
    pred_root = ISPY2_ROOT / "8_results_ispy2/01_predictions/ispy2_model" / train
    gt_root = ISPY2_ROOT / "2_nnUNet_ispy2/raw" / ISPY2_DATASET_DIR[train]
    return dict(train=train, eval_source="ispy2", eval_item=eval_item,
                noise_dir=pred_root / category_for(noise_id) / noise_id,
                real_dir=pred_root / category_for(real_id) / real_id,
                gt_dir=gt_root / f"labelsTs_{eval_item}", img_dir=gt_root / f"imagesTs_{eval_item}",
                pred_subdir=eval_item)


def duke_pair(train, eval_item, ladder_json):
    d = _ladder(ladder_json)
    noise_id = rung_run_id(d["run_keys"], "+voronoi (noise fill)", d["labels"])
    real_id = rung_run_id(d["run_keys"], "v26_6_2 (real fill)", d["labels"])
    pred_root = DUKE_ROOT / "8_results_duke-breast-mri/01_predictions/ispy2_model" / train
    gt_root = DUKE_ROOT / "2_nnUNet_duke-breast-mri/raw"
    return dict(train=train, eval_source="duke-breast-mri", eval_item=eval_item,
                noise_dir=pred_root / category_for(noise_id) / noise_id,
                real_dir=pred_root / category_for(real_id) / real_id,
                gt_dir=gt_root / f"labelsTs_{eval_item}", img_dir=gt_root / f"imagesTs_{eval_item}",
                pred_subdir=eval_item)


def build_pairs():
    pairs = [
        ispy2_pair("t1wce", "t1wce", ISPY2_ROOT / "8_results_ispy2/02_metrics/ispy2_model/t1wce/ablations/ladder_series.json"),
        ispy2_pair("t1wce", "t2w", ISPY2_ROOT / "8_results_ispy2/02_metrics/ispy2_model/t1wce/ablations/ladder_series.json"),
        duke_pair("t1wce", "precontrast_uni",
                  DUKE_ROOT / "8_results_duke-breast-mri/02_metrics/ispy2_model/t1wce/ablations/precontrast_uni/ladder_series.json"),
        ispy2_pair("t2w", "t2w", ISPY2_ROOT / "8_results_ispy2/02_metrics/ispy2_model/t2w/ablations/ladder_series.json"),
        ispy2_pair("t2w", "t1wce", ISPY2_ROOT / "8_results_ispy2/02_metrics/ispy2_model/t2w/ablations/ladder_series.json"),
        duke_pair("t2w", "t1wce_uni",
                  DUKE_ROOT / "8_results_duke-breast-mri/02_metrics/ispy2_model/t2w/ablations/t1wce_uni/ladder_series.json"),
        duke_pair("t2w", "precontrast_uni",
                  DUKE_ROOT / "8_results_duke-breast-mri/02_metrics/ispy2_model/t2w/ablations/precontrast_uni/ladder_series.json"),
    ]
    for p in pairs:
        p["in_domain"] = (p["eval_source"] == "ispy2" and p["eval_item"] == p["train"])
        p["same_contrast_cross_dataset"] = (p["eval_source"] == "duke-breast-mri"
                                             and p["eval_item"].startswith(p["train"]))
    return pairs


# ───────────────────────── per-image measures (all operate on a small cropped patch) ─────────────────────────
def step_d(region_vals, ring_vals):
    if len(region_vals) < 2 or len(ring_vals) < 2:
        return float("nan")
    vr, vs = float(region_vals.var()), float(ring_vals.var())
    denom = np.sqrt((vr + vs) / 2.0)
    return float(abs(region_vals.mean() - ring_vals.mean()) / denom) if denom > 0 else float("nan")


def step_auc(region_vals, ring_vals):
    n1, n2 = len(region_vals), len(ring_vals)
    if n1 < 2 or n2 < 2:
        return float("nan")
    try:
        res = mannwhitneyu(region_vals, ring_vals, alternative="two-sided")
    except ValueError:
        return float("nan")
    auc = float(res.statistic) / (n1 * n2)
    return max(auc, 1.0 - auc)


def ramp_R(img: np.ndarray, region: np.ndarray) -> float:
    n = int(region.sum())
    if n < MIN_VOX:
        return float("nan")
    vals = img[region]
    mu, sd = vals.mean(), vals.std()
    if sd <= 0:
        return float("nan")
    z = (vals - mu) / sd
    d_in = distance_transform_edt(region)[region]
    bins = np.clip(np.round(d_in), 1, 7).astype(int)
    uniq = np.unique(bins)
    if len(uniq) < 2:
        return float("nan")
    ss_tot = float(((z - z.mean()) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    ss_between = 0.0
    for b in uniq:
        m = bins == b
        nb = int(m.sum())
        if nb:
            ss_between += nb * (z[m].mean() - z.mean()) ** 2
    return float(ss_between / ss_tot)


def crop_bbox(mask: np.ndarray, shape, pad=PAD):
    """Bounding box of `mask` (already same shape as the full volume), padded and
    clipped to bounds. Returns a tuple of slices."""
    objs = find_objects(mask.astype(np.uint8))
    if not objs or objs[0] is None:
        return None
    sl = objs[0]
    out = []
    for s, dim in zip(sl, shape):
        lo = max(0, s.start - pad)
        hi = min(dim, s.stop + pad)
        out.append(slice(lo, hi))
    return tuple(out)


def region_ring(region: np.ndarray):
    d_in = distance_transform_edt(region)
    d_out = distance_transform_edt(~region)
    return (d_in > 1), ((d_out > 1) & (d_out <= D_OUT))


def dice_recall_precision(pred_full, gt_full, bbox):
    """TP/FP/FN/dice/recall/precision/volume MUST use the FULL (uncropped) pred and gt --
    a prediction can hallucinate false positives far from the GT region, and cropping to
    the GT bbox silently drops those, inflating dice/precision (bug found 2026-09-25: a
    breast case's crop-only dice of 0.18 vs the project's own official eval_all.csv dice
    of 0.058 for the identical case/fold/rung -- the crop was hiding real FPs). Only the
    ring-membership distance transform (for fp_share_ring, itself always local since
    D_OUT=5 << PAD=7) is computed on the small crop -- cheap and exact, since a voxel
    outside a bbox padded by PAD>D_OUT can never fall inside the ring."""
    G, P = gt_full == LABEL_ID, pred_full == LABEL_ID
    tp, fp = int((G & P).sum()), int((~G & P).sum())
    ng, npred = int(G.sum()), int(P.sum())
    Gc, Pc = G[bbox], P[bbox]
    d_out_map = distance_transform_edt(~Gc)
    ring = (d_out_map > 1) & (d_out_map <= D_OUT)
    fpm_c = (~Gc) & Pc
    fp_ring = int((fpm_c & ring).sum())
    return dict(
        dice=2 * tp / (ng + npred) if (ng + npred) else float("nan"),
        recall=tp / ng if ng else float("nan"),
        precision=tp / npred if npred else float("nan"),
        vol_ratio=npred / ng if ng else float("nan"),
        fp_share_ring=float(fp_ring / fp) if fp else float("nan"),
    )


def case_ids_for(gt_dir: Path):
    return sorted(f.name[:-7] for f in gt_dir.glob("*.nii.gz"))


def subsample(cases, n=MAX_CASES_PER_ROW, seed=SEED):
    if len(cases) <= n:
        return cases
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(cases), size=n, replace=False))
    return [cases[i] for i in idx]


# ───────────────────────── resumable shard I/O ─────────────────────────
def shard_paths(rank):
    return DATA_DIR / f"delta_shard{rank}.csv", DATA_DIR / f"vis_shard{rank}.csv", DATA_DIR / f"missing_shard{rank}.txt"


def done_cases(csv_path: Path, key_cols):
    if not csv_path.exists():
        return set()
    try:
        d = pd.read_csv(csv_path)
        return set(tuple(r) for r in d[key_cols].drop_duplicates().itertuples(index=False, name=None))
    except Exception:
        return set()


def append_rows(csv_path: Path, rows: list[dict], cols: list[str]):
    if not rows:
        return
    df = pd.DataFrame(rows)[cols]
    df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


def process_case(p, case) -> tuple[list[dict], dict | None, str | None]:
    """Returns (delta_rows, vis_row_or_None, missing_note_or_None) for one (pair, case)."""
    gt_path = p["gt_dir"] / f"{case}.nii.gz"
    gt_full = load(gt_path)
    region_full = gt_full == LABEL_ID
    if int(region_full.sum()) < MIN_VOX:
        return [], None, None
    bbox = crop_bbox(region_full, gt_full.shape)
    if bbox is None:
        return [], None, None
    gt = gt_full[bbox]
    region = gt == LABEL_ID

    delta_rows = []
    for fold in FOLDS:
        noise_f = p["noise_dir"] / fold / p["pred_subdir"] / f"{case}.nii.gz"
        real_f = p["real_dir"] / fold / p["pred_subdir"] / f"{case}.nii.gz"
        if not (noise_f.exists() and real_f.exists()):
            continue
        noise_pred_full, real_pred_full = load(noise_f), load(real_f)
        if noise_pred_full.shape != gt_full.shape or real_pred_full.shape != gt_full.shape:
            continue
        for rung, pred_full in (("noise", noise_pred_full), ("real", real_pred_full)):
            m = dice_recall_precision(pred_full, gt_full, bbox)
            delta_rows.append(dict(train=p["train"], eval_source=p["eval_source"], eval_item=p["eval_item"],
                                    case=case, fold=fold, rung=rung, in_domain=p["in_domain"],
                                    same_contrast_cross_dataset=p["same_contrast_cross_dataset"], **m))

    vis_row = None
    img_candidates = [p["img_dir"] / f"{case}_0000.nii.gz", p["img_dir"] / f"{case}.nii.gz"]
    img_path = next((c for c in img_candidates if c.exists()), None)
    if img_path is not None:
        img_full = load_img(img_path)
        if img_full.shape == gt_full.shape:
            img = img_full[bbox]
            R, ring = region_ring(region)
            n_R, n_ring = int(R.sum()), int(ring.sum())
            sd = sa = float("nan")
            if n_R >= MIN_VOX and n_ring >= MIN_VOX:
                sd, sa = step_d(img[R], img[ring]), step_auc(img[R], img[ring])
            r_val = ramp_R(img, region)
            vis_row = dict(train=p["train"], eval_source=p["eval_source"], eval_item=p["eval_item"], case=case,
                            in_domain=p["in_domain"], same_contrast_cross_dataset=p["same_contrast_cross_dataset"],
                            step_d=sd, step_auc=sa, ramp_R=r_val, gt_vox=int(region_full.sum()))

    missing = None
    if not delta_rows:
        missing = f"{p['train']}->{p['eval_source']}/{p['eval_item']} case={case}: no fold had both preds"
    return delta_rows, vis_row, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world-size", type=int, default=1)
    args = ap.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    delta_csv, vis_csv, missing_txt = shard_paths(args.rank)
    done_delta = done_cases(delta_csv, ["train", "eval_source", "eval_item", "case"])
    done_vis = done_cases(vis_csv, ["train", "eval_source", "eval_item", "case"])

    work = []
    for p in build_pairs():
        if not p["gt_dir"].is_dir():
            print(f"SKIP pair {p['train']}->{p['eval_source']}/{p['eval_item']}: no GT dir {p['gt_dir']}", flush=True)
            continue
        cases = subsample(case_ids_for(p["gt_dir"]))
        for c in cases:
            work.append((p, c))

    my_work = work[args.rank::args.world_size]
    print(f"rank {args.rank}/{args.world_size}: {len(my_work)}/{len(work)} (case,pair) items assigned", flush=True)

    t0 = time.time()
    n_done_now = 0
    with open(missing_txt, "a") as mf:
        for i, (p, case) in enumerate(my_work):
            key = (p["train"], p["eval_source"], p["eval_item"], case)
            need_delta = key not in done_delta
            need_vis = key not in done_vis
            if not need_delta and not need_vis:
                continue
            try:
                delta_rows, vis_row, missing = process_case(p, case)
            except Exception as e:  # noqa: BLE001 -- keep shard alive, log and move on
                mf.write(f"{key} EXCEPTION: {e}\n")
                mf.flush()
                continue
            if need_delta:
                append_rows(delta_csv, delta_rows, DELTA_COLS)
            if need_vis and vis_row is not None:
                append_rows(vis_csv, [vis_row], VIS_COLS)
            if missing:
                mf.write(missing + "\n")
                mf.flush()
            n_done_now += 1
            if (i + 1) % 10 == 0:
                dt = time.time() - t0
                print(f"rank {args.rank}: {i + 1}/{len(my_work)} items ({n_done_now} newly processed), "
                      f"{dt:.0f}s elapsed", flush=True)
    print(f"rank {args.rank}: DONE, {n_done_now} newly processed of {len(my_work)} assigned", flush=True)


if __name__ == "__main__":
    main()
