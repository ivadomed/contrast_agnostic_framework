#!/usr/bin/env python
"""
H7 summary: border-transition profiles vs the noise-fill/real-fill Dice swap.

============================== PRE-REGISTRATION (written BEFORE any shard was inspected;
outcomes have NOT been computed as of writing this docstring) ==============================

Hypothesis H7: the border transition matters. Noise-fill (K-means+Voronoi partition, each
region filled with a constant random mean + iid noise -> ALWAYS sharp region borders, optional
blur sigma<=0.8vox) trains a model that expects sharp borders. Real-fill (same partition, real
image under a random per-region affine, scale 0.5-2, random sign) trains a model that expects
the TRAINING contrast's real border profile.

Unit of analysis: per patient x eval-contrast x region R in {SNFH, RC, ET, NCR}, restricted to
the 70-patient GT-present population (patient_region_deltas.csv join gt_region_presence.csv,
gt_vox>0). Boundary profile per (patient, contrast, region): signed distance-to-boundary bins
(no d=0 bin — see compute_boundary_profiles.py docstring for why), z-scored intensity within
brain, normalized p(d)=(I(d)-I_out)/(I_in-I_out), W = width between p=0.8 and p=0.2 crossings
(first downward crossing scanning inside->outside; NaN if not found), STEP = |I_in-I_out| /
pooled-within-window std. PD(train,eval) = RMS(p_eval-p_train) over shared bins, same patient.
DW = |W_eval - W_train|.

PRE-REGISTERED PREDICTIONS (all one-sided, in the direction stated; alpha=0.05; each pooled
test uses the exact permutation test from ranking_within_train.py's kendall_S/pooled_p, rows =
(train, region), 3 eval points/row, pooled over the (up to) 12 rows; Spearman over all <=36
cells is supporting evidence only):
  P1  noise-fill Dice is higher where the EVAL border is sharp and strong:
        P1a: tau(dice_voronoi, -W_eval)   > 0   (pooled over rows)
        P1b: tau(dice_voronoi,  STEP_eval) > 0
      P1 HOLDS only if both P1a and P1b pooled p < 0.05.
  P2  the real-fill gain (delta = dice_realfill - dice_voronoi) is higher where the eval
      border profile matches the training one:
        P2a: tau(delta, -PD) > 0
        P2b: tau(delta, -DW) > 0
      P2 HOLDS only if both P2a and P2b pooled p < 0.05.
  P3  (edema/SNFH, t2w-trained model): W(t1n) < W(t1c), i.e. the t1n border is sharper than the
      t1c border for the SAME patients -- this is a pure contrast-physics claim (not tied to any
      training assignment), tested as a one-sided paired Wilcoxon signed-rank test on
      W(t1n)-W(t1c) per patient (alternative='less'), over patients with a valid W for both.

ROBUSTNESS (pre-registered, not exploratory): P1/P2 are re-run under two more (bin-range,
I_in/I_out-window) variants: robust_r4 = bins to +-4, windows (-4,-2)/(2,4); robust_r8 = bins to
+-8, windows (-8,-6)/(6,8) (primary = bins to +-6, windows (-6,-4)/(4,6), exactly the spec's own
window choice). A robustness variant that disagrees with the primary result is reported as-is,
not used to pick a "better" threshold post hoc.

SECONDARY, per-patient (between-patient variation; Holm-corrected across all 8; one-sided,
predicted sign is NEGATIVE for every test): Spearman(dice_voronoi, W_eval) and
Spearman(delta, PD), each in cells (t2w,t1n,SNFH), (t2w,t1c,SNFH), (t2f,t1n,SNFH),
(t1n,t2f,SNFH).

EXCLUSION RULES (pre-registered, applied identically regardless of what they do to any single
cell's result): a boundary-profile bin needs >=20 voxels or it's NaN; a patient's W/STEP/PD/DW
is NaN if the underlying normalization is degenerate (|I_in-I_out| pooled_std <= 1e-9, or a
crossing is missing); a CELL mean is reported but only feeds a pooled Kendall row if it has
>=10 valid patients; a Kendall ROW is used only if all 3 of its eval cells are valid for that
particular stat (P1a/P1b/P2a/P2b are checked for row-validity independently, since they use
different stats). n_rows actually used is reported for every pooled test — never silently
assumed to be 12.

Anything below the "==== OUTCOMES ====" marker in the printed/written report was computed AFTER
this docstring was written and committed to the shard-based data; no threshold, window, or
exclusion rule above was changed after seeing it.
================================================================================================

Usage:
  .venv/bin/python boundary_vs_fill_swap_summary.py                 # full run (needs all shards)
  .venv/bin/python boundary_vs_fill_swap_summary.py --plots-only    # replot from cached CSVs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from compute_boundary_profiles import (  # noqa: E402
    BINS, MAX_BIN, REGIONS, DATA_DIR,
    bin_stats_to_mean_var, normalize_profile, crossing_width, step_stat,
    pooled_mean_std, profile_distance,
)
from ranking_within_train import kendall_S, pooled_p  # noqa: E402

OUT = THIS_DIR.parent / "outputs"
TABLES, PLOTS = OUT / "tables", OUT / "plots"
TABLES.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

TRAINS = ("t1n", "t2w", "t2f")
CONTRASTS = ("t1n", "t1c", "t2w", "t2f")
MIN_CELL_PATIENTS = 10
KEY_CELLS = [  # for boundary_profiles_pairs.png, in this order
    ("t2w", "t1n", "SNFH"), ("t2w", "t1c", "SNFH"), ("t2w", "t2f", "SNFH"),
    ("t1n", "t2f", "SNFH"), ("t2f", "t1n", "SNFH"),
    ("t2w", "t1n", "RC"), ("t2f", "t1c", "RC"),
]
SECONDARY_CELLS = [("t2w", "t1n", "SNFH"), ("t2w", "t1c", "SNFH"),
                    ("t2f", "t1n", "SNFH"), ("t1n", "t2f", "SNFH")]
OUTCOME_COLOR = {"HELPS": "#2f7d6b", "HURTS": "#c0392b", "n.s.": "#888888"}

VARIANTS = {
    "primary": dict(bin_range=6, in_window=(-6, -4), out_window=(4, 6)),
    "robust_r4": dict(bin_range=4, in_window=(-4, -2), out_window=(2, 4)),
    "robust_r8": dict(bin_range=8, in_window=(-8, -6), out_window=(6, 8)),
}


def holm(pvals: dict) -> dict:
    """Standard Holm step-down correction. pvals: name -> raw p. Returns name -> adjusted p."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running_max = {}, 0.0
    for i, (name, p) in enumerate(items):
        val = min(1.0, (m - i) * p)
        running_max = max(running_max, val)
        adj[name] = running_max
    return adj


def one_sided_spearman(x, y, predicted_sign: int):
    """predicted_sign: +1 or -1. Returns (rho, p_one_sided)."""
    rho, p_two = spearmanr(x, y)
    if np.isnan(rho):
        return rho, float("nan")
    if np.sign(rho) == np.sign(predicted_sign) or rho == 0:
        return rho, p_two / 2.0
    return rho, 1.0 - p_two / 2.0


# ───────────────────────── data loading / merge ─────────────────────────
def load_bin_stats(shard_prefix="boundary_profile_shard") -> pd.DataFrame:
    shard_files = sorted(DATA_DIR.glob(f"{shard_prefix}*.csv"))
    if not shard_files:
        sys.exit(f"No shard CSVs found ({shard_prefix}*) in {DATA_DIR} — run the shards first")
    df = pd.concat([pd.read_csv(f) for f in shard_files], ignore_index=True)
    n_patients = df["patient"].nunique()
    if n_patients < 70:
        sys.exit(f"HARD FAIL: only {n_patients} patients in merged shards, expected >=70 "
                  f"(patient_region_deltas.csv population) — shards incomplete, not summarizing")
    df.to_csv(DATA_DIR / "boundary_bin_stats.csv", index=False)
    return df


def build_profiles(bin_df: pd.DataFrame):
    """Returns profiles[variant][(patient, region, contrast)] = dict(centers, p, i_in, i_out, W, STEP)."""
    profiles = {v: {} for v in VARIANTS}
    for (patient, region, contrast), g in bin_df.groupby(["patient", "region", "contrast"]):
        s = g.set_index("bin")
        for vname, cfg in VARIANTS.items():
            bins = [b for b in BINS if -cfg["bin_range"] <= b <= cfg["bin_range"]]
            centers = np.array(bins, dtype=float)
            n_arr = s["n"].reindex(bins).fillna(0).to_numpy()
            sz = s["sum_z"].reindex(bins).fillna(0).to_numpy()
            sz2 = s["sum_z2"].reindex(bins).fillna(0).to_numpy()
            mean_z, _ = bin_stats_to_mean_var(n_arr, sz, sz2)
            p, i_in, i_out = normalize_profile(centers, mean_z, cfg["in_window"], cfg["out_window"])
            w = crossing_width(centers, p)
            sel = (((centers >= cfg["in_window"][0]) & (centers <= cfg["in_window"][1])) |
                   ((centers >= cfg["out_window"][0]) & (centers <= cfg["out_window"][1])))
            _, pooled_std = pooled_mean_std(n_arr, sz, sz2, sel)
            step = step_stat(i_in, i_out, pooled_std)
            profiles[vname][(patient, region, contrast)] = dict(
                centers=centers, mean_z=mean_z, p=p, i_in=i_in, i_out=i_out, W=w, STEP=step)
    return profiles


def build_patient_table(profiles) -> pd.DataFrame:
    """Per-patient x train x eval x region rows with W_eval/STEP_eval/W_train/STEP_train/PD/DW
    for every variant, restricted to the GT-present population."""
    deltas = pd.read_csv(DATA_DIR / "patient_region_deltas.csv")
    present = pd.read_csv(DATA_DIR / "gt_region_presence.csv")
    present = present[present["gt_vox"] > 0][["case", "region"]]
    deltas = deltas.merge(present, on=["case", "region"])

    # cheap gate: cross-check against the already-published significance table (advisor's gate)
    sig = pd.read_csv(DATA_DIR / "region_fill_swap_significance.csv")
    sig = sig[sig["family"] == "OOD"]
    check = deltas.groupby(["train", "eval", "region"])["delta_dice"].mean().reset_index()
    check = check.merge(sig[["train", "eval", "region", "mean_delta_pts"]], on=["train", "eval", "region"])
    check["mine_pts"] = check["delta_dice"] * 100
    bad = check[(check["mine_pts"] - check["mean_delta_pts"]).abs() > 0.5]
    if len(bad):
        print("WARNING: cheap gate mismatch vs region_fill_swap_significance.csv:\n", bad)
    else:
        print(f"cheap gate OK: {len(check)} cells match region_fill_swap_significance.csv within 0.5 pts")

    for vname in VARIANTS:
        w_eval, step_eval, w_train, step_train, pd_col, dw_col = [], [], [], [], [], []
        prof = profiles[vname]
        for _, r in deltas.iterrows():
            pe = prof.get((r["case"], r["region"], r["eval"]))
            pt = prof.get((r["case"], r["region"], r["train"]))
            w_eval.append(pe["W"] if pe else float("nan"))
            step_eval.append(pe["STEP"] if pe else float("nan"))
            w_train.append(pt["W"] if pt else float("nan"))
            step_train.append(pt["STEP"] if pt else float("nan"))
            if pe and pt:
                pd_col.append(profile_distance(pe["centers"], pe["p"], pt["p"]))
                dw = abs(pe["W"] - pt["W"]) if np.isfinite(pe["W"]) and np.isfinite(pt["W"]) else float("nan")
                dw_col.append(dw)
            else:
                pd_col.append(float("nan"))
                dw_col.append(float("nan"))
        deltas[f"W_eval_{vname}"] = w_eval
        deltas[f"STEP_eval_{vname}"] = step_eval
        deltas[f"W_train_{vname}"] = w_train
        deltas[f"STEP_train_{vname}"] = step_train
        deltas[f"PD_{vname}"] = pd_col
        deltas[f"DW_{vname}"] = dw_col
    deltas.to_csv(DATA_DIR / "boundary_patient_table.csv", index=False)
    return deltas


def build_cells(patient_table: pd.DataFrame) -> pd.DataFrame:
    agg = {"dice_voronoi": "mean", "dice_realfill": "mean", "delta_dice": "mean", "case": "count"}
    for vname in VARIANTS:
        for stat in ("W_eval", "STEP_eval", "PD", "DW"):
            agg[f"{stat}_{vname}"] = "mean"
    patient_table = patient_table[patient_table["train"] != patient_table["eval"]]  # OOD cells only
    cells = patient_table.groupby(["train", "eval", "region"]).agg(agg).rename(
        columns={"case": "n_patients"}).reset_index()
    # per-stat valid patient counts (needed to null out under-powered cells independently)
    for vname in VARIANTS:
        for stat in ("W_eval", "STEP_eval", "PD", "DW"):
            col = f"{stat}_{vname}"
            counts = patient_table.groupby(["train", "eval", "region"])[col].apply(
                lambda s: int(s.notna().sum())).rename(f"n_{col}")
            cells = cells.merge(counts, on=["train", "eval", "region"])
            cells.loc[cells[f"n_{col}"] < MIN_CELL_PATIENTS, col] = np.nan
    cells.to_csv(DATA_DIR / "boundary_cells.csv", index=False)
    assert len(cells) == 36, f"expected 36 cells (3 trains x 3 evals x 4 regions), got {len(cells)}"
    return cells


# ───────────────────────── P1 / P2 pooled tests ─────────────────────────
def pooled_kendall_test(cells: pd.DataFrame, target_col: str, measure_col: str, sign: int):
    """Rows = (train, region), 3 eval points each. Only rows where target_col and measure_col
    are both non-null for all 3 evals are used. sign=+1 tests tau(target, measure)>0 (pass
    measure already flipped if the prediction is on -measure)."""
    S_total, n_rows = 0, 0
    row_details = []
    for (tr, region), g in cells.groupby(["train", "region"]):
        g = g.sort_values("eval")
        if len(g) != 3 or g[target_col].isna().any() or g[measure_col].isna().any():
            continue
        x = g[target_col].to_numpy()
        y = sign * g[measure_col].to_numpy()
        s = kendall_S(x, y)
        S_total += s
        n_rows += 1
        row_details.append(dict(train=tr, region=region, S=s, evals=",".join(g["eval"])))
    p = pooled_p(S_total, n_rows) if n_rows > 0 else float("nan")
    tau = S_total / (3 * n_rows) if n_rows > 0 else float("nan")
    return dict(S=S_total, n_rows=n_rows, tau=tau, p=p, rows=row_details)


def run_predictions(cells: pd.DataFrame, variant: str):
    out = {}
    out["P1a_dice_vs_negW"] = pooled_kendall_test(cells, "dice_voronoi", f"W_eval_{variant}", sign=-1)
    out["P1b_dice_vs_STEP"] = pooled_kendall_test(cells, "dice_voronoi", f"STEP_eval_{variant}", sign=+1)
    out["P2a_delta_vs_negPD"] = pooled_kendall_test(cells, "delta_dice", f"PD_{variant}", sign=-1)
    out["P2b_delta_vs_negDW"] = pooled_kendall_test(cells, "delta_dice", f"DW_{variant}", sign=-1)
    return out


# ───────────────────────── P3 ─────────────────────────
def run_p3(profiles):
    present = pd.read_csv(DATA_DIR / "gt_region_presence.csv")
    snfh_patients = present[(present["region"] == "SNFH") & (present["gt_vox"] > 0)]["case"].unique()
    prof = profiles["primary"]
    w_t1n, w_t1c = [], []
    for pid in snfh_patients:
        a, b = prof.get((pid, "SNFH", "t1n")), prof.get((pid, "SNFH", "t1c"))
        if a and b and np.isfinite(a["W"]) and np.isfinite(b["W"]):
            w_t1n.append(a["W"])
            w_t1c.append(b["W"])
    w_t1n, w_t1c = np.array(w_t1n), np.array(w_t1c)
    diff = w_t1n - w_t1c
    n = len(diff)
    if n < 5 or np.allclose(diff, 0):
        return dict(n=n, mean_W_t1n=np.mean(w_t1n) if n else float("nan"),
                    mean_W_t1c=np.mean(w_t1c) if n else float("nan"), stat=float("nan"), p=float("nan"))
    stat, p = wilcoxon(diff, alternative="less")
    return dict(n=n, mean_W_t1n=float(w_t1n.mean()), mean_W_t1c=float(w_t1c.mean()),
                stat=float(stat), p=float(p))


# ───────────────────────── secondary per-patient tests ─────────────────────────
def run_secondary(patient_table: pd.DataFrame):
    raw = {}
    for train, ev, region in SECONDARY_CELLS:
        sub = patient_table[(patient_table["train"] == train) & (patient_table["eval"] == ev) &
                             (patient_table["region"] == region)]
        for stat_name, xcol, ycol in [("dice_vs_W", "dice_voronoi", "W_eval_primary"),
                                        ("delta_vs_PD", "delta_dice", "PD_primary")]:
            s = sub[[xcol, ycol]].dropna()
            key = f"{train}->{ev} {region} {stat_name}"
            if len(s) >= 5:
                rho, p = one_sided_spearman(s[xcol].to_numpy(), s[ycol].to_numpy(), predicted_sign=-1)
                raw[key] = dict(n=len(s), rho=rho, p_raw=p)
            else:
                raw[key] = dict(n=len(s), rho=float("nan"), p_raw=float("nan"))
    p_raw = {k: v["p_raw"] for k, v in raw.items() if np.isfinite(v["p_raw"])}
    adj = holm(p_raw)
    for k, v in raw.items():
        v["p_holm"] = adj.get(k, float("nan"))
    return raw


# ───────────────────────── plots ─────────────────────────
CONTRAST_COLORS = {"t1n": "#4c72b0", "t1c": "#dd8452", "t2w": "#55a868", "t2f": "#c44e52"}


def plot_boundary_profiles(profiles):
    cfg = VARIANTS["primary"]
    bins = sorted(b for b in BINS if -cfg["bin_range"] <= b <= cfg["bin_range"])
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    for j, region in enumerate(REGIONS):
        for c in CONTRASTS:
            ps, raws = [], []
            for (patient, reg, contrast), d in profiles["primary"].items():
                if reg == region and contrast == c:
                    ps.append(d["p"])
                    raws.append(d["mean_z"])
            if not ps:
                continue
            ps, raws = np.array(ps), np.array(raws)
            med = np.nanmedian(ps, axis=0)
            q1, q3 = np.nanpercentile(ps, 25, axis=0), np.nanpercentile(ps, 75, axis=0)
            axes[0, j].plot(bins, med, color=CONTRAST_COLORS[c], label=c, linewidth=1.8)
            axes[0, j].fill_between(bins, q1, q3, color=CONTRAST_COLORS[c], alpha=0.15)
            raw_mean = np.nanmean(raws, axis=0)
            axes[1, j].plot(bins, raw_mean, color=CONTRAST_COLORS[c], label=c, linewidth=1.8)
        axes[0, j].set_title(region)
        axes[0, j].axhline(0.8, color="gray", linewidth=0.6, linestyle=":")
        axes[0, j].axhline(0.2, color="gray", linewidth=0.6, linestyle=":")
        axes[0, j].set_ylim(-0.3, 1.3)
        axes[1, j].set_xlabel("signed distance to boundary (voxels, inside<0)")
        axes[0, j].axvline(0, color="black", linewidth=0.5)
        axes[1, j].axvline(0, color="black", linewidth=0.5)
    axes[0, 0].set_ylabel("normalized p(d)\n(median, IQR band)")
    axes[1, 0].set_ylabel("raw z-scored I(d)\n(mean)")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Boundary intensity profiles by region and contrast (primary variant, bins ±6, "
                 "all GT-present patients pooled)")
    fig.tight_layout()
    fig.savefig(PLOTS / "boundary_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_boundary_profiles_pairs(profiles, sig: pd.DataFrame):
    cfg = VARIANTS["primary"]
    bins = sorted(b for b in BINS if -cfg["bin_range"] <= b <= cfg["bin_range"])
    n = len(KEY_CELLS)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows))
    axes = np.atleast_2d(axes)
    for idx, (train, ev, region) in enumerate(KEY_CELLS):
        ax = axes[idx // ncols, idx % ncols]
        for contrast, color, label in [(train, "black", f"train {train}"),
                                        (ev, CONTRAST_COLORS[ev], f"eval {ev}")]:
            ps = [d["p"] for (p_, reg, c), d in profiles["primary"].items() if reg == region and c == contrast]
            if not ps:
                continue
            med = np.nanmedian(np.array(ps), axis=0)
            ax.plot(bins, med, color=color, label=label, linewidth=1.8)
        row = sig[(sig["train"] == train) & (sig["eval"] == ev) & (sig["region"] == region)]
        delta = row["mean_delta_pts"].iloc[0] if len(row) else float("nan")
        direction = row["direction"].iloc[0] if len(row) else "?"
        ax.set_title(f"{train}→{ev} {region}\nΔ={delta:+.1f} pts ({direction})", fontsize=9)
        ax.axhline(0.8, color="gray", linewidth=0.5, linestyle=":")
        ax.axhline(0.2, color="gray", linewidth=0.5, linestyle=":")
        ax.set_ylim(-0.3, 1.3)
        ax.legend(fontsize=7, loc="lower left")
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")
    fig.suptitle("Training-contrast vs eval-contrast boundary profile, key cells (median over patients)")
    fig.tight_layout()
    fig.savefig(PLOTS / "boundary_profiles_pairs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(cells: pd.DataFrame, sig: pd.DataFrame):
    m = cells.merge(sig[["train", "eval", "region", "direction"]], on=["train", "eval", "region"], how="left")
    m["direction"] = m["direction"].fillna("n.s.")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for outcome, color in OUTCOME_COLOR.items():
        sub = m[m["direction"] == outcome]
        axes[0].scatter(sub["W_eval_primary"], sub["dice_voronoi"] * 100, color=color, label=outcome, s=40)
        axes[1].scatter(sub["PD_primary"], sub["delta_dice"] * 100, color=color, label=outcome, s=40)
    for _, r in m.iterrows():
        axes[0].annotate(f"{r['train']}→{r['eval']}", (r["W_eval_primary"], r["dice_voronoi"] * 100),
                          fontsize=6, alpha=0.7)
        axes[1].annotate(f"{r['train']}→{r['eval']}", (r["PD_primary"], r["delta_dice"] * 100),
                          fontsize=6, alpha=0.7)
    axes[0].set_xlabel("W_eval (transition width, voxels)")
    axes[0].set_ylabel("noise-fill (voronoi) Dice, %")
    axes[0].set_title("P1: sharper/stronger eval border -> higher noise-fill Dice?")
    axes[1].set_xlabel("PD (profile distance, train vs eval)")
    axes[1].set_ylabel("Δ Dice (real-fill − noise-fill), pts")
    axes[1].set_title("P2: matching border profile -> bigger real-fill gain?")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    axes[1].axhline(0, color="gray", linewidth=0.6)
    fig.suptitle("Boundary-profile statistics vs the noise-fill/real-fill Dice swap (all 36 cells)")
    fig.tight_layout()
    fig.savefig(PLOTS / "boundary_vs_fill_swap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ───────────────────────── report ─────────────────────────
def fmt_test(name, res):
    return (f"- **{name}**: pooled S={res['S']:+d} over n_rows={res['n_rows']} (of up to 12), "
            f"tau={res['tau']:+.3f}, one-sided p={res['p']:.4g} "
            f"-> {'SUPPORTED (p<0.05)' if np.isfinite(res['p']) and res['p'] < 0.05 else 'NOT supported'}")


def write_report(cells, preds_by_variant, p3, secondary, spearman36):
    lines = ["# H7: border-transition profiles vs noise-fill/real-fill Dice swap", ""]
    lines += ["## Pre-registered predictions — primary variant (bins +-6, windows (-6,-4)/(4,6))", ""]
    p = preds_by_variant["primary"]
    p1_hold = p["P1a_dice_vs_negW"]["p"] < 0.05 and p["P1b_dice_vs_STEP"]["p"] < 0.05
    p2_hold = p["P2a_delta_vs_negPD"]["p"] < 0.05 and p["P2b_delta_vs_negDW"]["p"] < 0.05
    lines.append(f"**P1 (noise-fill favors sharp/strong eval borders): "
                 f"{'HOLDS' if p1_hold else 'DOES NOT HOLD'}**")
    lines.append(fmt_test("P1a: tau(noisefill Dice, -W_eval) > 0", p["P1a_dice_vs_negW"]))
    lines.append(fmt_test("P1b: tau(noisefill Dice, STEP_eval) > 0", p["P1b_dice_vs_STEP"]))
    lines.append("")
    lines.append(f"**P2 (real-fill gain favors matching border profile): "
                 f"{'HOLDS' if p2_hold else 'DOES NOT HOLD'}**")
    lines.append(fmt_test("P2a: tau(delta, -PD) > 0", p["P2a_delta_vs_negPD"]))
    lines.append(fmt_test("P2b: tau(delta, -DW) > 0", p["P2b_delta_vs_negDW"]))
    lines.append("")
    p3_hold = np.isfinite(p3["p"]) and p3["p"] < 0.05
    lines.append(f"**P3 (edema, t2w-trained: W(t1n) < W(t1c)): {'HOLDS' if p3_hold else 'DOES NOT HOLD'}**")
    lines.append(f"- n={p3['n']} patients with valid W on both contrasts; "
                 f"mean W(t1n)={p3['mean_W_t1n']:.2f}, mean W(t1c)={p3['mean_W_t1c']:.2f}, "
                 f"Wilcoxon one-sided p={p3['p']:.4g}")
    lines.append("")
    lines.append("## Supporting: Spearman over the 36 cells (one-sided, predicted direction)")
    for k, (rho, pv) in spearman36.items():
        lines.append(f"- {k}: rho={rho:+.3f}, one-sided p={pv:.4g}")
    lines.append("")
    lines.append("## Robustness (pre-registered, not exploratory)")
    for vname in ("robust_r4", "robust_r8"):
        rp = preds_by_variant[vname]
        lines.append(f"**{vname}**:")
        for tname, res in rp.items():
            lines.append(f"  - {fmt_test(tname, res)}")
    lines.append("")
    lines.append("## Secondary per-patient tests (Holm-corrected across all 8, one-sided negative)")
    lines.append("| cell | n | rho | p_raw | p_holm |")
    lines.append("|---|--:|--:|--:|--:|")
    for k, v in secondary.items():
        lines.append(f"| {k} | {v['n']} | {v['rho']:+.3f} | {v['p_raw']:.4g} | {v['p_holm']:.4g} |")
    lines.append("")
    lines.append("## Per-cell table (SNFH first)")
    lines.append("")
    lines.append("| train | eval | region | n | dice_voronoi | dice_realfill | Delta | W_eval | STEP_eval | PD | DW |")
    lines.append("|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    order = {"SNFH": 0, "RC": 1, "ET": 2, "NCR": 3}
    for _, r in cells.assign(_o=cells["region"].map(order)).sort_values(["_o", "train", "eval"]).iterrows():
        lines.append(f"| {r['train']} | {r['eval']} | {r['region']} | {r['n_patients']} | "
                     f"{100*r['dice_voronoi']:.1f} | {100*r['dice_realfill']:.1f} | {100*r['delta_dice']:+.1f} | "
                     f"{r['W_eval_primary']:.2f} | {r['STEP_eval_primary']:.2f} | "
                     f"{r['PD_primary']:.3f} | {r['DW_primary']:.2f} |")
    (TABLES / "boundary_vs_fill_swap.md").write_text("\n".join(lines))
    print("\n".join(lines[:40]))
    print(f"... full report written to {TABLES / 'boundary_vs_fill_swap.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots-only", action="store_true")
    ap.add_argument("--shard-prefix", default="boundary_profile_shard")
    args = ap.parse_args()

    cells_csv = DATA_DIR / "boundary_cells.csv"
    patient_csv = DATA_DIR / "boundary_patient_table.csv"
    if args.plots_only and cells_csv.exists() and patient_csv.exists():
        cells = pd.read_csv(cells_csv)
        patient_table = pd.read_csv(patient_csv)
        bin_df = pd.read_csv(DATA_DIR / "boundary_bin_stats.csv")
        profiles = build_profiles(bin_df)
    else:
        bin_df = load_bin_stats(args.shard_prefix)
        profiles = build_profiles(bin_df)
        patient_table = build_patient_table(profiles)
        cells = build_cells(patient_table)

    preds_by_variant = {v: run_predictions(cells, v) for v in VARIANTS}
    p3 = run_p3(profiles)
    secondary = run_secondary(patient_table)

    spearman36 = {}
    m = cells.dropna(subset=["W_eval_primary", "dice_voronoi"])
    spearman36["dice_voronoi vs -W_eval"] = one_sided_spearman(m["dice_voronoi"], -m["W_eval_primary"], +1)
    m = cells.dropna(subset=["STEP_eval_primary", "dice_voronoi"])
    spearman36["dice_voronoi vs STEP_eval"] = one_sided_spearman(m["dice_voronoi"], m["STEP_eval_primary"], +1)
    m = cells.dropna(subset=["PD_primary", "delta_dice"])
    spearman36["delta_dice vs -PD"] = one_sided_spearman(m["delta_dice"], -m["PD_primary"], +1)
    m = cells.dropna(subset=["DW_primary", "delta_dice"])
    spearman36["delta_dice vs -DW"] = one_sided_spearman(m["delta_dice"], -m["DW_primary"], +1)

    sig = pd.read_csv(DATA_DIR / "region_fill_swap_significance.csv")
    sig = sig[sig["family"] == "OOD"]

    write_report(cells, preds_by_variant, p3, secondary, spearman36)
    plot_boundary_profiles(profiles)
    plot_boundary_profiles_pairs(profiles, sig)
    plot_scatter(cells, sig)
    print("Plots written to", PLOTS)


if __name__ == "__main__":
    main()
