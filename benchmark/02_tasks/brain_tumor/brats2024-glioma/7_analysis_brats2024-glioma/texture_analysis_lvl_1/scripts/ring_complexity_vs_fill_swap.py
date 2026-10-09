#!/usr/bin/env python
"""
Does the texture complexity of the SURROUNDINGS (ring 1-5 vox) explain where real-fill helps or hurts? (Paul, 2026-10-07)
Inputs: outputs/data/ring_complexity_shard*.csv (compute_ring_complexity.py), patient_region_deltas.csv (val000, 4 arms),
region_fill_swap_significance.csv.
PRE-REGISTERED (before reading any number):
  Hypothesis H: more complex surroundings on the EVAL image make the texture cue misleading -> real-fill gains less /
  loses more. Prediction: within a cell, Spearman rho(ring complexity_eval, delta) < 0.
  Unit = patient within cell (the design that sidesteps the contrast-identity confound). Primary test, per feature:
  the 48 OOD cells' within-cell rho (cells with n>=20), one-sided Wilcoxon signed-rank of rho vs 0 (cells as units,
  H1: median rho < 0); Holm over the 5 ring features. Secondary: same with the eval-minus-train gap (same patient,
  co-registered) and with the region's own complexity; cell-level: Spearman(cell mean, cell delta) over the 48 cells and sign
  accuracy on the 15 Holm-significant cells of 'noise wins iff ring complexity_eval above its (region) median' vs
  always-helps (12/15). Exploratory: failure cells vs success cells compared on their within-cell rho.
Output: outputs/tables/ring_complexity_vs_fill_swap.md, outputs/data/ring_complexity_cells.csv
"""
from __future__ import annotations
import glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon, mannwhitneyu
OUT = Path(__file__).resolve().parent.parent / "outputs"; D = OUT / "data"; T = OUT / "tables"
FEATS = ("dog_slope", "ms_ratio", "vario_slope", "fd", "lstd_cv", "lstd_ent")
MIN_N = 20


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m)
    run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(run, 1.0)
    return adj


def main():
    f = pd.concat([pd.read_csv(x) for x in sorted(glob.glob(str(D / "ring_complexity_shard*.csv")))]).drop_duplicates(["patient", "contrast", "region"])
    d = pd.read_csv(D / "patient_region_deltas.csv"); d = d[d["train"] != d["eval"]]
    sig = pd.read_csv(D / "region_fill_swap_significance.csv"); sig = sig[sig.family == "OOD"]
    fe = f.rename(columns={"patient": "case", "contrast": "eval"}).drop(columns=["n_R", "n_ring"])
    ft = f.rename(columns={"patient": "case", "contrast": "train"}).drop(columns=["n_R", "n_ring"])
    m = d.merge(fe, on=["case", "eval", "region"]).merge(ft, on=["case", "train", "region"], suffixes=("_eval", "_train"))
    for part in ("ring", "R"):
        for k in FEATS:
            m[f"{part}_{k}_gap"] = m[f"{part}_{k}_eval"] - m[f"{part}_{k}_train"]
    cols = [f"{part}_{k}_{v}" for part in ("ring", "R") for k in FEATS for v in ("eval", "gap")]
    rows = []
    for (tr, ev, rg), g in m.groupby(["train", "eval", "region"]):
        if len(g) < MIN_N:
            continue
        r = dict(train=tr, eval=ev, region=rg, n=len(g), delta=100 * g.delta_dice.mean())
        for c in cols:
            x = g[c].values; ok = np.isfinite(x)
            r[f"rho_{c}"] = spearmanr(x[ok], g.delta_dice.values[ok])[0] if ok.sum() >= MIN_N else np.nan
            r[f"mean_{c}"] = float(np.nanmean(x))
        rows.append(r)
    cells = pd.DataFrame(rows).merge(sig[["train", "eval", "region", "significant", "direction", "p_holm"]], on=["train", "eval", "region"])
    cells.to_csv(D / "ring_complexity_cells.csv", index=False)
    L = ["# Surroundings texture complexity vs the fill-swap effect (BraTS, val000, 4 training arms)", "",
         f"{len(cells)} OOD cells with n>={MIN_N} patients; unit = patient within cell. Pre-registered: within-cell "
         "Spearman rho(ring complexity on the EVAL image, delta real-noise Dice) < 0 (complex surroundings -> real-fill gains less). "
         "Test = one-sided Wilcoxon signed-rank of the per-cell rho (cells as units), Holm over the 5 ring features (fd is a "
         "deterministic transform of vario_slope and is not counted twice).", ""]
    prim = [f"ring_{k}_eval" for k in ("dog_slope", "ms_ratio", "vario_slope", "lstd_cv", "lstd_ent")]

    def block(title, names, one_sided=True):
        out = [f"## {title}", "", "| feature | cells | median rho | cells rho<0 | Wilcoxon p (H1: rho<0) | p Holm | cell-level Spearman(mean, delta) (p) | median rho in failure cells / success cells |", "|---|--:|--:|--:|--:|--:|--:|--:|"]
        ps, recs = [], []
        for c in names:
            r = cells[f"rho_{c}"].dropna()
            p = wilcoxon(r, alternative="less").pvalue if len(r) >= 6 else np.nan
            ps.append(p)
            cl = spearmanr(cells[f"mean_{c}"], cells["delta"], nan_policy="omit")
            fr = cells[cells.direction == "HURTS"][f"rho_{c}"].median(); su = cells[cells.direction == "HELPS"][f"rho_{c}"].median()
            recs.append((c, len(r), r.median(), int((r < 0).sum()), p, cl, fr, su))
        ph = holm(ps)
        for (c, n, med, neg, p, cl, fr, su), h in zip(recs, ph):
            out.append(f"| {c} | {n} | {med:+.3f} | {neg}/{n} | {p:.3g} | {h:.3g} | {cl[0]:+.2f} ({cl[1]:.2g}) | {fr:+.3f} / {su:+.3f} |")
        return out + [""]
    L += block("PRIMARY: ring complexity on the eval image", prim)
    L += block("Secondary: ring complexity gap (eval - train, same patient)", [f"ring_{k}_gap" for k in ("dog_slope", "ms_ratio", "vario_slope", "lstd_cv", "lstd_ent")])
    L += block("Secondary: the region's own complexity (eval image)", [f"R_{k}_eval" for k in ("dog_slope", "ms_ratio", "vario_slope", "lstd_cv", "lstd_ent")])
    L += block("Secondary: region complexity gap (eval - train)", [f"R_{k}_gap" for k in ("dog_slope", "ms_ratio", "vario_slope", "lstd_cv", "lstd_ent")])
    # cell-level sign rule on the significant cells
    S = cells[cells.significant.astype(bool)].copy()
    L += ["## Cell-level sign rule on the Holm-significant cells", "", f"{len(S)} significant cells, {int((S.delta < 0).sum())} failures; "
          f"'always helps' = {int((S.delta > 0).sum())}/{len(S)}. Rule: noise wins iff the cell's mean ring complexity (eval) is above the median of its region's OOD cells.", "",
          "| feature | sig acc | failures caught | false alarms |", "|---|--:|--:|--:|"]
    for c in prim:
        med = cells.groupby("region")[f"mean_{c}"].transform("median")
        hi = cells[f"mean_{c}"] > med
        pred = np.where(hi, -1, 1); y = np.sign(cells.delta.values); ssel = cells.significant.astype(bool).values
        acc = (pred[ssel] == y[ssel]).mean(); fail = (y < 0) & ssel
        L.append(f"| {c} | {acc:.2f} | {int(((pred < 0) & fail).sum())}/{int(fail.sum())} | {int(((pred < 0) & ssel & ~fail).sum())} |")
    L += ["", "## Per-cell detail (significant cells)", "", "| cell | delta | call | " + " | ".join(f"rho {k}" for k in ("dog_slope", "ms_ratio", "vario_slope", "lstd_cv", "lstd_ent")) + " | mean ring fd eval / train |", "|---|--:|---|" + "--:|" * 6]
    for r in S.sort_values("delta").itertuples():
        L.append(f"| {r.train}->{r.eval} {r.region} | {r.delta:+.1f} | {r.direction} | " + " | ".join(f"{getattr(r, f'rho_ring_{k}_eval'):+.2f}" for k in ("dog_slope", "ms_ratio", "vario_slope", "lstd_cv", "lstd_ent")) + f" | {getattr(r, 'mean_ring_fd_eval'):.2f} / {getattr(r, 'mean_ring_fd_eval') - getattr(r, 'mean_ring_fd_gap'):.2f} |")
    L += ["", "## Mean ring complexity per contrast x region (eval image)", "", "```", m.groupby(["eval", "region"])[[f"ring_{k}_eval" for k in FEATS]].mean().round(3).to_string(), "```"]
    (T / "ring_complexity_vs_fill_swap.md").write_text("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
