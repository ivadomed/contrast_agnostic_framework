#!/usr/bin/env python
"""
Paul's hypothesis (2026-10-07, after the "train me" session): real-fill beats noise-fill on EASY images; on HARD images
noise-fill wins for LARGE regions while real-fill still wins for SMALL regions.
Unit = patient within an OOD cell (train, eval, region), BraTS val000, 4 training arms, 48 cells. Delta = real-fill minus
noise-fill Dice (per patient, mean over folds 0-2; patient_region_deltas.csv).
Difficulty, two independent definitions:
  D_cons = 1 - mean Dice of the 4 OTHER headline methods of the same training arm (baseline, auglab_default, synthseg_EM,
           srcsm; synthseg_noEM excluded, it collapses on lesion tasks) on that (eval contrast, case, region), folds 0-2.
  D_vis  = 1 - visibility AUC of the region vs its 1-5 vox ring on the eval image (step_affine_per_patient_step.csv).
Size = log10 GT voxels of the region (gt_region_presence.csv).
PRE-REGISTERED (written before any number):
  P1  easy half (D below the cell median): mean Delta > 0.
  P2  hard & large (D above, size above the cell medians): mean Delta < 0.
  P3  hard & small: mean Delta > 0.
  P4  interaction: within hard cases, Delta decreases with size (per-cell Spearman rho(size, Delta | hard) < 0, Wilcoxon
      over cells); and the pooled fixed-effect regression Delta ~ D + size + D*size has a negative D*size coefficient.
  Quadrants are defined within cell (medians of that cell's patients) so no cell-level contrast effect leaks in.
  Reported for D_cons (primary) and D_vis (secondary), all 48 cells and the 15 Holm-significant cells.
Output: outputs/tables/difficulty_size_vs_fill_swap.md, outputs/data/difficulty_size_rows.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from compute_cross_contrast_ngf import METRICS_ROOT, FOLDS  # noqa: E402
D = THIS.parent / "outputs" / "data"; T = THIS.parent / "outputs" / "tables"
OTHERS = {"baseline": "nnUNet_brats2024-glioma_{c}_baseline_", "auglab_default": "auglab_brats2024-glioma_{c}_auglab_default_",
          "synthseg_EM": "auglab_brats2024-glioma_{c}_synthseg_EM_", "srcsm": "auglab_brats2024-glioma_{c}_srcsm_"}
REG = ("NCR", "SNFH", "ET", "RC")


def consensus(train):
    frames = []
    for m, pat in OTHERS.items():
        dirs = [d for d in (METRICS_ROOT / train).iterdir() if d.name.startswith(pat.format(c=train)) and not d.name.endswith(("_final", "_srcmatch"))]
        assert len(dirs) == 1, (train, m, dirs)
        for f in FOLDS:
            df = pd.read_csv(dirs[0] / f / "eval_all.csv"); df["method"] = m; frames.append(df)
    df = pd.concat(frames); df = df[df.label.isin(REG)]
    return df.groupby(["group", "case", "label"])["dice"].mean().rename("other_dice").reset_index().rename(columns={"group": "eval", "label": "region"})


def quadrant_table(df, dcol, title):
    q = df.copy()
    g = q.groupby(["train", "eval", "region"])
    q["hard"] = q[dcol] > g[dcol].transform("median"); q["large"] = q["logv"] > g["logv"].transform("median")
    L = [f"### {title}", "", "| quadrant | n rows | mean Δ (pts) | median Δ | frac Δ>0 | Wilcoxon p (two-sided, rows) | cells with mean Δ<0 |", "|---|--:|--:|--:|--:|--:|--:|"]
    for name, sel in (("easy", ~q.hard), ("hard", q.hard), ("easy & small", ~q.hard & ~q.large), ("easy & large", ~q.hard & q.large),
                      ("hard & small", q.hard & ~q.large), ("hard & large", q.hard & q.large)):
        s = q[sel]; d = 100 * s.delta_dice; cm = s.groupby(["train", "eval", "region"]).delta_dice.mean()
        p = wilcoxon(d).pvalue if len(d) > 10 else np.nan
        L.append(f"| {name} | {len(s)} | {d.mean():+.2f} | {d.median():+.2f} | {(d > 0).mean():.2f} | {p:.2g} | {int((cm < 0).sum())}/{len(cm)} |")
    # P4: within hard rows, per-cell Spearman(size, Delta)
    rh = []
    for k, s in q[q.hard].groupby(["train", "eval", "region"]):
        if len(s) >= 10:
            rh.append(spearmanr(s.logv, s.delta_dice)[0])
    rh = np.array(rh); pw = wilcoxon(rh, alternative="less").pvalue if len(rh) >= 6 else np.nan
    # pooled fixed-effects regression with within-cell standardised D and size
    z = q.copy()
    for c in (dcol, "logv"):
        z[c + "_z"] = g[c].transform(lambda v: (v - v.mean()) / (v.std() + 1e-9))
    z["inter"] = z[dcol + "_z"] * z["logv_z"]
    cells = pd.get_dummies(z.train + ">" + z["eval"] + ":" + z.region, drop_first=False).astype(float)
    X = np.column_stack([z[dcol + "_z"], z["logv_z"], z["inter"], cells.values]); y = 100 * z.delta_dice.values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ beta
    n, k = X.shape; s2 = (res ** 2).sum() / (n - k); cov = s2 * np.linalg.pinv(X.T @ X); se = np.sqrt(np.diag(cov))[:3]
    from scipy.stats import t as tdist
    tt = beta[:3] / se; pp = 2 * tdist.sf(np.abs(tt), n - k)
    L += ["", f"P4 within hard rows: per-cell Spearman rho(size, Δ) median {np.median(rh):+.3f}, {int((rh < 0).sum())}/{len(rh)} cells negative, Wilcoxon p (H1 rho<0) = {pw:.3g}.",
          f"Pooled OLS with cell fixed effects (Δ in pts; D and size standardised within cell): D {beta[0]:+.2f} (p={pp[0]:.2g}), size {beta[1]:+.2f} (p={pp[1]:.2g}), D×size {beta[2]:+.2f} (p={pp[2]:.2g}); n={n} rows.", ""]
    return L, q


def main():
    d = pd.read_csv(D / "patient_region_deltas.csv"); d = d[d.train != d["eval"]]
    v = pd.read_csv(D / "gt_region_presence.csv"); d = d.merge(v, on=["case", "region"]); d["logv"] = np.log10(d.gt_vox.clip(lower=1))
    cons = pd.concat([consensus(tr).assign(train=tr) for tr in ("t1n", "t1c", "t2w", "t2f")])
    d = d.merge(cons, on=["train", "eval", "case", "region"], how="left"); d["D_cons"] = 1 - d.other_dice
    st = pd.read_csv(D / "step_affine_per_patient_step.csv").rename(columns={"patient": "case", "contrast": "eval"})[["case", "eval", "region", "step_auc"]]
    d = d.merge(st, on=["case", "eval", "region"], how="left"); d["D_vis"] = 1 - d.step_auc
    sig = pd.read_csv(D / "region_fill_swap_significance.csv"); sig = sig[sig.family == "OOD"][["train", "eval", "region", "significant", "direction"]]
    d = d.merge(sig, on=["train", "eval", "region"]); d.to_csv(D / "difficulty_size_rows.csv", index=False)
    L = ["# Difficulty x region size vs the fill-swap effect (BraTS, val000, 48 OOD cells, patient unit)", "",
         "Hypothesis (Paul): real-fill wins on easy images; on hard images noise-fill wins for large regions, real-fill still for small ones. "
         "Quadrants split at each cell's own medians. Δ = real-fill − noise-fill Dice per patient (pts).", "",
         f"Rows: {len(d)}; D_cons available for {d.D_cons.notna().sum()}, D_vis for {d.D_vis.notna().sum()}.", ""]
    for dcol, lab in (("D_cons", "difficulty = 1 − mean Dice of the 4 other methods (consensus)"), ("D_vis", "difficulty = 1 − visibility AUC on the eval image")):
        dd = d.dropna(subset=[dcol])
        L.append(f"## {lab}")
        for title, sel in (("all 48 OOD cells", dd), ("15 Holm-significant cells", dd[dd.significant.astype(bool)]),
                           ("3 failure cells (t2w→t1n SNFH, t2f→t1n SNFH, t2f→t1c RC)", dd[dd.direction == "HURTS"]), ("12 success cells", dd[dd.direction == "HELPS"])):
            out, _ = quadrant_table(sel, dcol, title); L += out
    # per failure cell detail: Δ by size tertile and by difficulty tertile
    L += ["## Failure cells, Δ by size and difficulty tertiles (D_cons)", "", "| cell | size tertile: small / mid / large (mean Δ, n) | difficulty tertile: easy / mid / hard (mean Δ, n) |", "|---|---|---|"]
    for k, s in d[d.direction == "HURTS"].groupby(["train", "eval", "region"]):
        s = s.dropna(subset=["D_cons"]); st_ = pd.qcut(s.logv, 3, labels=["small", "mid", "large"]); dt = pd.qcut(s.D_cons, 3, labels=["easy", "mid", "hard"])
        a = s.groupby(st_, observed=True).delta_dice.agg(["mean", "size"]); b = s.groupby(dt, observed=True).delta_dice.agg(["mean", "size"])
        L.append(f"| {k[0]}→{k[1]} {k[2]} | " + " / ".join(f"{100 * r['mean']:+.1f} ({int(r['size'])})" for _, r in a.iterrows()) + " | " + " / ".join(f"{100 * r['mean']:+.1f} ({int(r['size'])})" for _, r in b.iterrows()) + " |")
    (T / "difficulty_size_vs_fill_swap.md").write_text("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
