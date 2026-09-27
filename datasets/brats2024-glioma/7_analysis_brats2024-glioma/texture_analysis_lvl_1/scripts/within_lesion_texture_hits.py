#!/usr/bin/env python
"""
Within-lesion texture vs prediction-hit analysis — BraTS2024-glioma, t1n-trained models,
edema (SNFH, label 2) on the REAL FLAIR (t2f) eval contrast.

PRE-REGISTERED PREDICTION (written before running any analysis):
  Within-patient, inside real GT edema voxels on the real FLAIR image, the NOISE-FILL
  (Voronoi rung4) model's hit rate for label 2 DECREASES with local texture magnitude T
  more steeply than the REAL-FILL (v26_6_2 train050_val100, rung5) model's hit rate does,
  controlling for depth-to-border and local intensity. I.e. beta_T(noise) < beta_T(real),
  tested with a paired Wilcoxon signed-rank test across the 70 eval patients, plus each
  beta_T tested against 0 across patients.

No image edits are made anywhere in this script — FLAIR images are real acquired volumes;
"noise-fill" / "real-fill" refers only to which trained model produced the segmentation
prediction being scored.

Method per patient:
  - Load real FLAIR image + GT label (via load_patient, reused from compute_cross_contrast_ngf.py).
  - edema_mask = GT label == 2.
  - T (local texture) = std of highpass(FLAIR) in a 5x5x5 window, computed on the whole
    volume then restricted to edema voxels (so the window can see context just outside
    edema, matching compute_region_surround_texture.py's own windowing choice).
  - depth = Euclidean distance transform to the edema mask's own border (voxels units).
  - I = z-scored real FLAIR intensity (z'd within the patient's edema voxels).
  - hit = predicted label == 2 at that voxel, majority vote over folds 0/1/2 (>=2 of 3),
    computed separately for the noise-fill (auglab baseline_kmeans_label_remap_voronoi)
    and real-fill (nnUNet v26_6_2 train050_val100) predictions.
  - Per patient, per model: logistic regression hit ~ z(T) + z(depth) + z(I), extract
    coefficient beta_T. Patients where edema is too small (<MIN_VOX) or hit is
    constant (all-hit / all-miss, coefficient undefined) are skipped for that model.
  - Across patients: paired Wilcoxon (stat_tests.wilcoxon_p) on beta_T(noise) vs
    beta_T(real) (only patients with both), and one-sample Wilcoxon (vs 0, via
    scipy directly since stat_tests.wilcoxon_p is paired-only) for each model's beta_T.
  - Pooled hit-rate by texture quintile (pooled across all edema voxels from all
    patients) for both models, for the table/plot.

Usage (inside a Slurm CPU job, never on the login node):
  .venv/bin/python within_lesion_texture_hits.py --sanity
  .venv/bin/python within_lesion_texture_hits.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.ndimage import distance_transform_edt, uniform_filter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[5]
DS_ROOT = PROJECT_ROOT / "datasets" / "brats2024-glioma"
DELTAS_CSV = (
    DS_ROOT
    / "7_analysis_brats2024-glioma"
    / "texture_analysis_lvl_1"
    / "outputs"
    / "data"
    / "patient_region_deltas.csv"
)

import importlib.util


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ngf_mod = _load_module("compute_cross_contrast_ngf", SCRIPT_DIR / "compute_cross_contrast_ngf.py")
surround_mod = _load_module("compute_region_surround_texture", SCRIPT_DIR / "compute_region_surround_texture.py")

load_patient = ngf_mod.load_patient
region_masks = ngf_mod.region_masks
highpass = surround_mod.highpass
HP_SIGMA = surround_mod.HP_SIGMA

sys.path.insert(0, str(DS_ROOT.parents[0] / "00_commun_scripts" / "00_00_utils"))
from stat_tests import wilcoxon_p  # noqa: E402

SCRATCH = Path(__import__("os").environ["SCRATCH"])
PRED_ROOT = SCRATCH / "brats_ladder_preds" / "t1n"
NOISE_RUN = "auglab/brats2024-glioma_t1n_baseline_kmeans_label_remap_voronoi_20260730_200711"
REAL_RUN = "nnUNet/brats2024-glioma_t1n_v26_6_2_train050_val100_20260730_200711"
FOLDS = ("fold0", "fold1", "fold2")
EDEMA_LABEL = 2
MIN_VOX = 50
WIN = 5  # 5x5x5 window for local texture std

OUT_DIR = SCRIPT_DIR.parents[0] / "outputs"
TABLE_DIR = OUT_DIR / "tables"
PLOT_DIR = OUT_DIR / "plots"
LOG_DIR = OUT_DIR.parents[0] / "outputs" / "logs"
for d in (TABLE_DIR, PLOT_DIR):
    d.mkdir(parents=True, exist_ok=True)


def local_texture_std(highpassed: np.ndarray, win: int = WIN) -> np.ndarray:
    """std of `highpassed` in a win^3 window, via E[x^2]-E[x]^2 with uniform_filter."""
    m1 = uniform_filter(highpassed, size=win, mode="nearest")
    m2 = uniform_filter(highpassed ** 2, size=win, mode="nearest")
    var = np.clip(m2 - m1 ** 2, 0, None)
    return np.sqrt(var)


def load_prediction_majority(run_dir: Path, case: str, shape) -> np.ndarray | None:
    """Majority (>=2 of 3 folds) predicted-edema-label mask for one case, or None if any fold missing."""
    import nibabel as nib

    votes = np.zeros(shape, dtype=np.int8)
    n_found = 0
    for fold in FOLDS:
        f = run_dir / fold / "t2f" / f"{case}.nii.gz"
        if not f.exists():
            continue
        pred = np.asarray(nib.load(str(f)).dataobj).round().astype(np.int64)
        if pred.shape != shape:
            log.warning("%s %s: shape mismatch %s vs %s, skipping fold", case, fold, pred.shape, shape)
            continue
        votes += (pred == EDEMA_LABEL).astype(np.int8)
        n_found += 1
    if n_found < len(FOLDS):
        return None
    return (votes >= 2).astype(np.int8)


def zscore(x: np.ndarray) -> np.ndarray:
    s = x.std()
    if s < 1e-8:
        return np.zeros_like(x)
    return (x - x.mean()) / s


def fit_beta_T(hit: np.ndarray, T: np.ndarray, depth: np.ndarray, I: np.ndarray) -> float | None:
    """Logistic regression hit ~ zT + zdepth + zI, return beta_T. None if undefined."""
    if hit.sum() == 0 or hit.sum() == len(hit):
        return None
    zT, zD, zI = zscore(T), zscore(depth), zscore(I)
    X = np.column_stack([np.ones_like(zT), zT, zD, zI])
    y = hit.astype(np.float64)
    # Newton-Raphson IRLS for stability (no sklearn dependency assumed available).
    beta = np.zeros(X.shape[1])
    for _ in range(50):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        W = p * (1 - p)
        W = np.clip(W, 1e-6, None)
        grad = X.T @ (y - p)
        H = -(X.T * W) @ X
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            return None
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < 1e-6:
            beta = beta_new
            break
        beta = beta_new
    if not np.all(np.isfinite(beta)):
        return None
    return float(beta[1])


def run_sanity() -> int:
    """Synthetic check: a model whose hit prob decreases with T should get beta_T < 0."""
    rng = np.random.default_rng(0)
    n = 2000
    T = rng.normal(size=n)
    depth = rng.normal(size=n)
    I = rng.normal(size=n)
    logit = -1.5 * T + 0.1 * depth
    p = 1 / (1 + np.exp(-logit))
    hit = (rng.uniform(size=n) < p).astype(np.int8)
    beta = fit_beta_T(hit, T, depth, I)
    log.info("sanity beta_T=%.3f (expect clearly negative)", beta)
    assert beta is not None and beta < -0.5, f"sanity check failed: beta_T={beta}"
    print("SANITY OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()
    if args.sanity:
        return run_sanity()

    device = torch.device("cpu")
    cases = sorted(pd.read_csv(DELTAS_CSV)["case"].unique().tolist())
    log.info("Loaded %d eval cases", len(cases))

    noise_dir = PRED_ROOT / NOISE_RUN
    real_dir = PRED_ROOT / REAL_RUN

    per_patient_rows = []
    quintile_pool = {"noise": [], "real": []}  # list of (T, hit) arrays

    for i, case in enumerate(cases):
        try:
            vols, label = load_patient(case, device)
        except FileNotFoundError as e:
            log.warning("skip %s: %s", case, e)
            continue
        flair = vols["t2f"].cpu().numpy().astype(np.float64)
        label_np = label.cpu().numpy()
        edema = label_np == EDEMA_LABEL
        n_edema = int(edema.sum())
        if n_edema < MIN_VOX:
            log.info("skip %s: edema too small (%d vox)", case, n_edema)
            continue

        brain = flair > 0
        hp = highpass(flair, brain)
        T_full = local_texture_std(hp, WIN)

        depth_full = distance_transform_edt(edema)  # distance to nearest border-outside voxel, inside mask
        # distance_transform_edt on boolean array gives distance to nearest False voxel (=border) for True voxels.

        T = T_full[edema]
        depth = depth_full[edema]
        I = flair[edema]

        shape = flair.shape
        hit_noise_full = load_prediction_majority(noise_dir, case, shape)
        hit_real_full = load_prediction_majority(real_dir, case, shape)
        if hit_noise_full is None or hit_real_full is None:
            log.warning("skip %s: missing prediction fold(s)", case)
            continue
        hit_noise = hit_noise_full[edema]
        hit_real = hit_real_full[edema]

        beta_noise = fit_beta_T(hit_noise, T, depth, I)
        beta_real = fit_beta_T(hit_real, T, depth, I)

        per_patient_rows.append(
            {
                "case": case,
                "n_edema_vox": n_edema,
                "hitrate_noise": float(hit_noise.mean()),
                "hitrate_real": float(hit_real.mean()),
                "beta_T_noise": beta_noise,
                "beta_T_real": beta_real,
            }
        )
        quintile_pool["noise"].append((T, hit_noise))
        quintile_pool["real"].append((T, hit_real))

        if (i + 1) % 10 == 0:
            log.info("processed %d/%d", i + 1, len(cases))

    df = pd.DataFrame(per_patient_rows)
    df.to_csv(TABLE_DIR / "within_lesion_texture_hits_per_patient.csv", index=False)
    log.info("Per-patient rows: %d", len(df))

    # Paired test: beta_T(noise) - beta_T(real), only patients with both defined.
    both = df.dropna(subset=["beta_T_noise", "beta_T_real"])
    diff = both["beta_T_noise"].to_numpy() - both["beta_T_real"].to_numpy()
    p_paired = wilcoxon_p(both["beta_T_noise"].to_numpy(), both["beta_T_real"].to_numpy())

    def one_sample_wilcoxon(x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        if len(x) < 1 or not np.any(x != 0):
            return float("nan")
        try:
            return float(stats.wilcoxon(x, alternative="two-sided", zero_method="wilcox").pvalue)
        except ValueError:
            return float("nan")

    p_noise_vs0 = one_sample_wilcoxon(df["beta_T_noise"].to_numpy())
    p_real_vs0 = one_sample_wilcoxon(df["beta_T_real"].to_numpy())

    n_neg_diff = int((diff < 0).sum())
    n_pos_diff = int((diff > 0).sum())

    # Pooled quintile hit rates.
    def pooled_quintiles(pool):
        Ts = np.concatenate([t for t, _ in pool])
        hits = np.concatenate([h for _, h in pool])
        qs = pd.qcut(Ts, 5, labels=False, duplicates="drop")
        out = []
        for q in sorted(np.unique(qs)):
            m = qs == q
            out.append((q, float(hits[m].mean()), int(m.sum())))
        return out

    q_noise = pooled_quintiles(quintile_pool["noise"])
    q_real = pooled_quintiles(quintile_pool["real"])

    # ---- Table ----
    lines = []
    lines.append("# Within-lesion (edema) texture vs prediction-hit — t1n-trained, real FLAIR eval")
    lines.append("")
    lines.append("Pre-registered prediction: noise-fill hit rate decreases with texture T more than "
                  "real-fill's, controlling for depth-to-border and intensity (beta_T(noise) < beta_T(real)).")
    lines.append("")
    lines.append(f"- N patients with usable data (both models): {len(both)} / {len(cases)} eval cases")
    lines.append(f"- Paired Wilcoxon, beta_T(noise) vs beta_T(real): p = {p_paired:.4g} "
                 f"(n_noise<real: {n_neg_diff}, n_noise>real: {n_pos_diff})")
    lines.append(f"- Mean beta_T(noise) = {df['beta_T_noise'].mean():.4f} "
                 f"(one-sample Wilcoxon vs 0: p = {p_noise_vs0:.4g})")
    lines.append(f"- Mean beta_T(real)  = {df['beta_T_real'].mean():.4f} "
                 f"(one-sample Wilcoxon vs 0: p = {p_real_vs0:.4g})")
    lines.append("")
    lines.append("## Pooled hit rate by texture quintile (Q0=lowest texture .. Q4=highest)")
    lines.append("")
    lines.append("| quintile | hit rate (noise-fill) | n (noise) | hit rate (real-fill) | n (real) |")
    lines.append("|---|---|---|---|---|")
    for (qn, hn, nn_), (qr, hr, nr_) in zip(q_noise, q_real):
        lines.append(f"| Q{qn} | {hn:.4f} | {nn_} | {hr:.4f} | {nr_} |")
    lines.append("")
    lines.append(f"Full per-patient table: `outputs/tables/within_lesion_texture_hits_per_patient.csv`")
    (TABLE_DIR / "within_lesion_texture_hits.md").write_text("\n".join(lines) + "\n")
    log.info("Wrote table.")

    # ---- Plot ----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    xs = list(range(5))
    axes[0].plot(xs, [h for _, h, _ in q_noise], "o-", label="noise-fill (Voronoi)", color="#d62728")
    axes[0].plot(xs, [h for _, h, _ in q_real], "o-", label="real-fill (v26_6_2)", color="#1f77b4")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([f"Q{i}" for i in xs])
    axes[0].set_xlabel("local texture quintile (low -> high)")
    axes[0].set_ylabel("pooled edema-label hit rate")
    axes[0].set_title("Hit rate vs texture, pooled voxels")
    axes[0].legend()

    axes[1].scatter(both["beta_T_real"], both["beta_T_noise"], alpha=0.6)
    lo = min(both["beta_T_real"].min(), both["beta_T_noise"].min())
    hi = max(both["beta_T_real"].max(), both["beta_T_noise"].max())
    axes[1].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[1].set_xlabel("beta_T (real-fill)")
    axes[1].set_ylabel("beta_T (noise-fill)")
    axes[1].set_title(f"Per-patient beta_T, n={len(both)}\npaired Wilcoxon p={p_paired:.3g}")

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "within_lesion_texture_hits.png", dpi=150)
    log.info("Wrote plot.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
