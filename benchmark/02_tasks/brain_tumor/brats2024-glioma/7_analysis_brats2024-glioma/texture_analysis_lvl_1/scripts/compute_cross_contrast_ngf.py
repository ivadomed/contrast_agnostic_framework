#!/usr/bin/env python
"""
Cross-contrast texture similarity (NGF) — BraTS2024-glioma pilot.

Question this answers: for the SAME patient and SAME anatomical region, how similar is the
local texture (gradient-orientation field) between two REAL, co-registered contrasts (e.g. real
T1n vs real T2w)? This is deliberately NOT the source-vs-generated NGF used in open-ms's
compute_ngf_texture.py (which measures how well a SYNTHESIS method preserves one image's own
texture) — here both sides of every comparison are real acquired volumes of the same patient,
already co-registered to a common grid by the BraTS preprocessing pipeline (checked below, not
assumed).

Motivation: the causal-ablation ladder's rung4->5 step (Voronoi noise-fill -> PALETTE real-fill)
sometimes HURTS cross-contrast Dice — e.g. a t2w-trained model evaluated on t1n gets WORSE with
real-fill (pooled Delta Dice = -4.59), while every eval direction OUT OF t1n gets much better with
real-fill. Hypothesis: real-fill teaches the network texture specific to the training contrast;
this helps generalization when the eval contrast's true local texture agrees with the training
contrast's, and can hurt when it doesn't. NGF(A,B) is symmetric, so a "low NGF -> hurts" test
can't reproduce a direction-dependent sign flip by itself — we therefore report, per region and
per contrast pair:
  - ngf_all:    plain symmetric mean over the region (chance floor ~1/3, ceiling 1 — see
                Haber & Modersitzki 2006 formula reused from open-ms/.../compute_ngf_texture.py)
  - ngf_edge_a: gated on contrast A's own edges (top 50% |grad_A| in the region) — "at the
                places that matter in A, does B's gradient agree in direction"
  - ngf_edge_b: symmetric counterpart, gated on B's edges
The directional test against the ladder uses ngf_edge_<TRAIN>, i.e. gate on whichever contrast
was the *training* contrast for that ladder — "at the training contrast's own texture-defining
edges, does the eval contrast agree" — since that is the texture the real-fill augmentation
actually bakes into training.

Also pulls the REAL per-region rung4->5 (voronoi noise-fill -> v26_6_2 real-fill) Dice deltas
straight from each training contrast's existing ablation eval_all.csv files (not hardcoded),
grouped by label (NCR/SNFH/ET — RC is dropped, see below), and prints them side by side with the
single-patient NGF numbers for the same (train, eval, region) triples.

Usage (run inside a Slurm CPU job — NOT on the login node, NOT on TamIA per user instruction):
  python compute_cross_contrast_ngf.py --sanity
  python compute_cross_contrast_ngf.py --output-dir <dir>
"""
from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[7]
DS_ROOT = PROJECT_ROOT / "benchmark" / "02_tasks" / "brain_tumor" / "brats2024-glioma"
BIDS_ROOT = DS_ROOT / "1_BIDS_brats2024-glioma" / "glioma-brain-brats2024"
LABELS_DIR = (DS_ROOT / "2_nnUNet_brats2024-glioma" / "raw"
              / "Dataset051_BraTS2024GliomaT1n" / "labelsTr")
METRICS_ROOT = DS_ROOT / "8_results_brats2024-glioma" / "02_metrics" / "brats2024_glioma_model"
OUT_DIR_DEFAULT = (DS_ROOT / "7_analysis_brats2024-glioma" / "texture_analysis_lvl_1" / "outputs")

# BIDS filename suffix per contrast key (all four live under sub-<ID>/anat/, same grid).
CONTRAST_SUFFIX = {
    "t1n": "T1w.nii",
    "t1c": "ce-gadolinium_T1w.nii",
    "t2w": "T2w.nii",
    "t2f": "FLAIR.nii",
}
LABEL_IDS = {"NCR": 1, "SNFH": 2, "ET": 3, "RC": 4}
MIN_REGION_VOX = 50

# Ladder rung dirs: rung4 = voronoi noise-fill, rung5 = v26_6_2 real-fill (PALETTE alone).
# Hardcoded (verified to exist, 2026-09-24) rather than regex-discovered: the t2w real-fill run
# (nnUNet_..._20260620_125217) predates the project's 3-fold policy and still has a fold3/ dir
# sitting next to fold0-2 — a glob-based "find any matching run dir" approach silently pulled it
# in (280 rows instead of 210 for a 3-fold rung), which is a REAL bug this pilot hit: it changed
# a headline per-region delta (t2w->t1n SNFH) by roughly 3x vs the folds-0-2-only ground truth
# that ladder_summary.md itself reports. FOLDS is hardcoded to 0-2 below and enforced everywhere.
FOLDS = ("fold0", "fold1", "fold2")
RUNG_DIRS = {
    # 2026-10-07: rung 5 -> the val000 retrains (checkpoint_best chosen on REAL validation, like rung 4;
    # the earlier *_val100_* runs picked it on synthetic validation and are retired, see the project notes
    # "Rung 5 ... are val000 runs (2026-10-06)"). t1c (4th training contrast, ladder 06_33) added.
    "t1n": {
        "voronoi": "ablations/auglab_brats2024-glioma_t1n_baseline_kmeans_label_remap_voronoi_20260730_200711",
        "realfill": "ablations/auglab_brats2024-glioma_t1n_v26_6_2_train050_val000_20261005_222227",
    },
    "t1c": {
        "voronoi": "ablations/auglab_brats2024-glioma_t1c_baseline_kmeans_label_remap_voronoi_20260921_140000",
        "realfill": "ablations/auglab_brats2024-glioma_t1c_v26_6_2_train050_val000_20261005_222406",
    },
    "t2w": {
        "voronoi": "ablations/auglab_brats2024-glioma_t2w_baseline_kmeans_label_remap_voronoi_20260805_020659",
        "realfill": "ablations/auglab_brats2024-glioma_t2w_v26_6_2_train050_val000_20261005_222300",
    },
    "t2f": {
        "voronoi": "ablations/auglab_brats2024-glioma_t2f_baseline_kmeans_label_remap_voronoi_20260917_094019",
        "realfill": "ablations/auglab_brats2024-glioma_t2f_v26_6_2_train050_val000_20261005_222333",
    },
}
TRAIN_CONTRASTS = ("t1n", "t1c", "t2w", "t2f")
# Published ladder_series.json pooled-OOD Dice (rung4 -> rung5), for the sanity gate below
# (2026-10-06 regeneration with the val000 rung 5; "dice"[3], "dice"[4] of each json).
PUBLISHED_OOD_DELTA_DICE = {"t1n": 45.78 - 39.81, "t1c": 43.19 - 30.46, "t2w": 42.08 - 40.95, "t2f": 30.67 - 31.92}

EPS = 1e-5
EDGE_PCTL = 50


# ───────────────────────── NGF math (ported from open-ms compute_ngf_texture.py, unchanged) ────
def _sobel_kernels(device, dtype):
    d = torch.tensor([1., 0., -1.], device=device, dtype=dtype)
    s = torch.tensor([1., 2., 1.], device=device, dtype=dtype)

    def outer3(a, b, c):
        return torch.einsum('i,j,k->ijk', a, b, c)

    kz = outer3(d, s, s)[None, None]
    ky = outer3(s, d, s)[None, None]
    kx = outer3(s, s, d)[None, None]
    return kz, ky, kx


def gradient_3d(x: torch.Tensor):
    kz, ky, kx = _sobel_kernels(x.device, x.dtype)
    xp = F.pad(x[None, None], (1, 1, 1, 1, 1, 1), mode="replicate")
    gz = F.conv3d(xp, kz)[0, 0]
    gy = F.conv3d(xp, ky)[0, 0]
    gx = F.conv3d(xp, kx)[0, 0]
    return gz, gy, gx


def ngf_map(a, b, eps=EPS):
    az, ay, ax = gradient_3d(a)
    bz, by, bx = gradient_3d(b)
    dot = az * bz + ay * by + ax * bx
    a_norm2 = az * az + ay * ay + ax * ax + eps ** 2
    b_norm2 = bz * bz + by * by + bx * bx + eps ** 2
    sim = (dot * dot) / (a_norm2 * b_norm2)
    return sim, (az * az + ay * ay + ax * ax), (bz * bz + by * by + bx * bx)


def ngf_scores(a, b, mask, edge_pctl=EDGE_PCTL, eps=EPS):
    """Returns (ngf_all, ngf_edge_a, ngf_edge_b, n) over `mask`."""
    sim, a_g2, b_g2 = ngf_map(a, b, eps)
    v_sim, va_g2, vb_g2 = sim[mask], a_g2[mask], b_g2[mask]
    n = int(mask.sum().item())
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    ngf_all = float(v_sim.mean())

    def edge_gated(g2):
        thresh = torch.quantile(g2, edge_pctl / 100.0)
        sel = g2 >= thresh
        return float(v_sim[sel].mean()) if int(sel.sum()) > 0 else float("nan")

    return ngf_all, edge_gated(va_g2), edge_gated(vb_g2), n


def run_sanity(device) -> int:
    torch.manual_seed(0)
    D = 44

    def boxf(x, w):
        return F.avg_pool3d(x[None, None], w, 1, w // 2, count_include_pad=False)[0, 0]

    a = boxf(torch.randn(D, D, D, device=device), 5)
    a = (a - a.min()) / (a.max() - a.min() + 1e-7)
    mask = torch.ones(D, D, D, dtype=torch.bool, device=device)
    mask[:3] = mask[-3:] = mask[:, :3] = mask[:, -3:] = mask[:, :, :3] = mask[:, :, -3:] = False

    cases = {
        "identity": a.clone(),
        "gamma": a.clamp_min(1e-4) ** 2.0,
        "inverted": 1.0 - a,
        "noise": torch.rand(D, D, D, device=device),
    }
    ok = True
    print(f"{'case':10s} {'ngf_all':>10s} {'edge_a':>10s} {'edge_b':>10s}")
    for name, b in cases.items():
        all_, ea, eb, _ = ngf_scores(a, b, mask)
        print(f"{name:10s} {all_:10.3f} {ea:10.3f} {eb:10.3f}")
        if name in ("identity", "gamma", "inverted"):
            ok &= all_ > 0.95
        if name == "noise":
            ok &= 0.20 < all_ < 0.50
    print(f"\nSANITY {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


# ───────────────────────── data loading ─────────────────────────
def load_vol(path: Path, device) -> torch.Tensor:
    img = nib.load(str(path))
    return torch.as_tensor(np.asarray(img.dataobj, dtype=np.float32), device=device), img.affine


def bids_case_id(patient_id: str) -> str:
    return f"sub-{patient_id}"


def load_patient(patient_id: str, device):
    """Loads all 4 contrasts + label volume for one patient, verifying they share one grid."""
    anat_dir = BIDS_ROOT / bids_case_id(patient_id) / "anat"
    vols, affines = {}, {}
    for key, suffix in CONTRAST_SUFFIX.items():
        f = anat_dir / f"{bids_case_id(patient_id)}_{suffix}"
        if not f.exists():
            raise FileNotFoundError(f)
        vols[key], affines[key] = load_vol(f, device)
    label_f = LABELS_DIR / f"{patient_id}.nii.gz"
    label_img = nib.load(str(label_f))
    label = torch.as_tensor(np.asarray(label_img.dataobj).round().astype(np.int64), device=device)

    shapes = {k: tuple(v.shape) for k, v in vols.items()}
    shapes["label"] = tuple(label.shape)
    if len(set(shapes.values())) != 1:
        raise ValueError(f"{patient_id}: shape mismatch across contrasts/label: {shapes}")
    for k, aff in affines.items():
        if not np.allclose(aff, affines["t1n"], atol=1e-3):
            raise ValueError(f"{patient_id}: affine mismatch t1n vs {k}")
    return vols, label


def region_masks(label: torch.Tensor, t1n: torch.Tensor):
    masks = {}
    for name, lid in LABEL_IDS.items():
        masks[name] = (label == lid)
    masks["tumor_core"] = (label == LABEL_IDS["NCR"]) | (label == LABEL_IDS["ET"])
    masks["whole_tumor"] = label > 0
    fg = t1n > (0.10 * torch.quantile(t1n[t1n > 0], 0.99))
    masks["healthy"] = fg & (label == 0)
    return masks


def load_rung_eval(train_contrast: str, rung: str) -> pd.DataFrame:
    """Loads eval_all.csv for exactly FOLDS (0-2), never picking up a stray fold3/ dir that
    predates the 3-fold policy (see RUNG_DIRS comment — this was a real bug in an earlier
    version of this script)."""
    run_dir = METRICS_ROOT / train_contrast / RUNG_DIRS[train_contrast][rung]
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    frames = []
    for fold in FOLDS:
        csv = run_dir / fold / "eval_all.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        df["fold"] = fold
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no fold0-2 eval_all.csv under {run_dir}")
    return pd.concat(frames, ignore_index=True)


def verify_against_published(train_contrast: str, voronoi: pd.DataFrame, realfill: pd.DataFrame) -> bool:
    """Gate: this script's own pooled-OOD delta must match ladder_summary.md's published number
    (computed independently by significance_from_config.py's pipeline) before any per-region or
    per-patient number derived here is trusted."""
    ood_contrasts = [c for c in ("t1n", "t1c", "t2w", "t2f") if c != train_contrast]
    v = voronoi[voronoi["group"].isin(ood_contrasts)]["dice"].mean() * 100
    r = realfill[realfill["group"].isin(ood_contrasts)]["dice"].mean() * 100
    mine = r - v
    theirs = PUBLISHED_OOD_DELTA_DICE[train_contrast]
    ok = abs(mine - theirs) < 1.0
    log.info("VERIFY train=%s pooled-OOD Delta Dice: this script=%.2f  published=%.2f  %s",
              train_contrast, mine, theirs, "PASS" if ok else "FAIL")
    return ok


def patient_region_deltas(train_contrast: str) -> pd.DataFrame:
    """Per-patient, per-eval-contrast, per-region Dice delta (real-fill - noise-fill), averaged
    over folds 0-2. This is the per-patient target the NGF correlation is tested against."""
    voronoi = load_rung_eval(train_contrast, "voronoi")
    realfill = load_rung_eval(train_contrast, "realfill")
    if not verify_against_published(train_contrast, voronoi, realfill):
        log.warning("train=%s: pooled delta does not match ladder_summary.md — "
                     "per-patient numbers below may still be wrong, treat with caution", train_contrast)

    def case_means(df):
        return (df[df["label"].isin(("NCR", "SNFH", "ET", "RC"))]
                .groupby(["group", "case", "label"])["dice"].mean().reset_index())

    v_means, r_means = case_means(voronoi), case_means(realfill)
    merged = v_means.merge(r_means, on=["group", "case", "label"], suffixes=("_voronoi", "_realfill"))
    merged = merged.dropna(subset=["dice_voronoi", "dice_realfill"])
    merged["delta_dice"] = merged["dice_realfill"] - merged["dice_voronoi"]
    merged["train"] = train_contrast
    merged = merged.rename(columns={"group": "eval", "label": "region"})
    return merged[["train", "eval", "case", "region", "dice_voronoi", "dice_realfill", "delta_dice"]]


NGF_ROW_COLS = ["patient", "region", "contrast_a", "contrast_b", "n_vox",
                "ngf_all", "ngf_edge_a", "ngf_edge_b"]


def shard_csv_path(data_dir: Path, rank: int) -> Path:
    return data_dir / f"cross_contrast_ngf_shard{rank}.csv"


def already_done_patients(shard_csv: Path) -> set:
    if not shard_csv.exists():
        return set()
    try:
        return set(pd.read_csv(shard_csv)["patient"].unique())
    except Exception:  # noqa: BLE001 — corrupt/partial file from a killed job, redo it
        return set()


def compute_ngf_shard(unique_patients, rank, world_size, device, data_dir):
    """Computes NGF for this rank's slice of patients, appending to a per-rank CSV as each
    patient finishes (not buffered in memory to one final write) — so a killed/preempted job
    loses at most the in-flight patient, and a rerun with the same --rank/--world-size skips
    whatever is already in the shard CSV instead of recomputing it."""
    my_patients = unique_patients[rank::world_size]
    shard_csv = shard_csv_path(data_dir, rank)
    done = already_done_patients(shard_csv)
    todo = [p for p in my_patients if p not in done]
    log.info("rank %d/%d: %d patients assigned, %d already done, %d to do",
              rank, world_size, len(my_patients), len(done), len(todo))

    pairs = list(itertools.combinations(CONTRAST_SUFFIX.keys(), 2))
    n_ok, n_fail = 0, 0
    write_header = not shard_csv.exists()
    for i, patient_id in enumerate(todo):
        try:
            vols, label = load_patient(patient_id, device)
        except (FileNotFoundError, ValueError) as e:
            n_fail += 1
            log.warning("SKIP %s: %s", patient_id, e)
            continue
        masks = region_masks(label, vols["t1n"])
        rows = []
        for region, mask in masks.items():
            n = int(mask.sum().item())
            if n < MIN_REGION_VOX:
                continue
            for a, b in pairs:
                ngf_all, ngf_edge_a, ngf_edge_b, nvox = ngf_scores(vols[a], vols[b], mask)
                rows.append(dict(patient=patient_id, region=region, contrast_a=a, contrast_b=b,
                                  n_vox=nvox, ngf_all=ngf_all, ngf_edge_a=ngf_edge_a,
                                  ngf_edge_b=ngf_edge_b))
        pd.DataFrame(rows, columns=NGF_ROW_COLS).to_csv(
            shard_csv, mode="a", header=write_header, index=False)
        write_header = False
        n_ok += 1
        del vols, label
        if (i + 1) % 5 == 0:
            log.info("  rank %d: %d/%d done (%d ok, %d skipped)", rank, i + 1, len(todo), n_ok, n_fail)
    log.info("rank %d: finished, %d ok / %d failed this run (shard total now %d patients)",
              rank, n_ok, n_fail, len(already_done_patients(shard_csv)))


def merge_and_correlate(data_dir: Path):
    """Concatenates all shard CSVs, joins against the (already-verified) per-patient ladder
    deltas, and reports the per-(train,eval,region) cross-patient correlation. Run this once
    ALL shards have finished."""
    shard_files = sorted(data_dir.glob("cross_contrast_ngf_shard*.csv"))
    if not shard_files:
        sys.exit(f"No shard CSVs found in {data_dir} — run the shards first")
    ngf_df = pd.concat([pd.read_csv(f) for f in shard_files], ignore_index=True)
    ngf_csv = data_dir / "cross_contrast_ngf_per_patient.csv"
    ngf_df.to_csv(ngf_csv, index=False)
    log.info("Merged %d shard file(s) -> %d NGF rows -> %s", len(shard_files), len(ngf_df), ngf_csv)

    delta_frames = [patient_region_deltas(c) for c in TRAIN_CONTRASTS]
    delta_df = pd.concat(delta_frames, ignore_index=True)
    delta_csv = data_dir / "patient_region_deltas.csv"
    delta_df.to_csv(delta_csv, index=False)
    log.info("Wrote %d per-patient delta rows -> %s", len(delta_df), delta_csv)

    # ── merge: for each (train, eval, region, patient) with BOTH a delta and an NGF value, join
    # on the unordered contrast pair. ngf_edge_a/b collapsed to within ~0.02 of ngf_all in the
    # single-patient pilot (gating on train vs eval contrast made no real difference), so ngf_all
    # (symmetric) is the primary reported metric here, not a train-gated directional one.
    ngf_pairs = ngf_df.set_index(["patient", "region", "contrast_a", "contrast_b"])["ngf_all"].to_dict()

    def lookup_ngf(patient, train, eval_c, region):
        for a, b in ((train, eval_c), (eval_c, train)):
            v = ngf_pairs.get((patient, region, a, b))
            if v is not None:
                return v
        return None

    delta_df["ngf_all"] = [lookup_ngf(r["case"], r["train"], r["eval"], r["region"])
                            for _, r in delta_df.iterrows()]
    merged = delta_df.dropna(subset=["ngf_all"])
    merged_csv = data_dir / "ngf_vs_ladder_per_patient.csv"
    merged.to_csv(merged_csv, index=False)
    log.info("Wrote %d merged (patient-level) rows -> %s", len(merged), merged_csv)

    from scipy.stats import spearmanr
    print("\n=== Per-(train,eval,region) cross-patient correlation: NGF(train,eval) vs real-fill Dice delta ===")
    print(f"{'train':6s}{'eval':6s}{'region':8s}{'n':>5s}{'mean_delta':>12s}{'spearman_rho':>14s}{'p':>10s}")
    cell_rows = []
    for (train, ev, region), g in merged.groupby(["train", "eval", "region"]):
        n = len(g)
        mean_delta = g["delta_dice"].mean()
        if n >= 8:
            rho, p_val = spearmanr(g["ngf_all"], g["delta_dice"])
        else:
            rho, p_val = float("nan"), float("nan")
        cell_rows.append(dict(train=train, eval=ev, region=region, n=n, mean_delta_dice=mean_delta,
                               spearman_rho=rho, p_value=p_val))
        print(f"{train:6s}{ev:6s}{region:8s}{n:5d}{mean_delta:12.4f}{rho:14.3f}{p_val:10.3f}"
              if n >= 8 else f"{train:6s}{ev:6s}{region:8s}{n:5d}{mean_delta:12.4f}{'n<8, skipped':>14s}")
    cell_df = pd.DataFrame(cell_rows)
    cell_csv = data_dir / "ngf_vs_ladder_cell_correlations.csv"
    cell_df.to_csv(cell_csv, index=False)
    log.info("Wrote %d cell-level correlation rows -> %s", len(cell_df), cell_csv)


# ───────────────────────── main ─────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=OUT_DIR_DEFAULT)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--limit-patients", type=int, default=None,
                    help="cap unique patients loaded (debug/smoke-test only)")
    p.add_argument("--rank", type=int, default=0, help="shard index, 0-based")
    p.add_argument("--world-size", type=int, default=1, help="number of parallel shards")
    p.add_argument("--merge", action="store_true",
                    help="skip NGF computation; merge existing shard CSVs and run the correlation")
    p.add_argument("--sanity", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    if args.sanity:
        sys.exit(run_sanity(device))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge_and_correlate(data_dir)
        return

    log.info("Loading per-patient rung4->5 (noise-fill -> real-fill) region Dice deltas "
              "from eval_all.csv (folds 0-2 only), with a sanity gate against ladder_summary.md...")
    delta_frames = [patient_region_deltas(c) for c in TRAIN_CONTRASTS]
    delta_df = pd.concat(delta_frames, ignore_index=True)
    unique_patients = sorted(delta_df["case"].unique())
    if args.limit_patients:
        unique_patients = unique_patients[: args.limit_patients]
    log.info("%d unique patients total across all shards", len(unique_patients))

    compute_ngf_shard(unique_patients, args.rank, args.world_size, device, data_dir)


if __name__ == "__main__":
    main()
