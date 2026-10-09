#!/usr/bin/env python
"""
Pooled comparison: per-region fill-swap ablation outcome (region_fill_swap_significance.csv) vs.
feature-level (learned cross-contrast synthesis) containment R2 (synthesis_r2_per_patient.csv).
Mirrors containment_vs_fill_swap_summary.py's structure (same directory, correlation-ratio eta^2
version) -- read that file's docstring for the shared hypothesis framing.

R2(train | eval) = mean over patients of R2(source=eval, target=train) -- can a CNN trained to
predict the TRAIN contrast from the EVAL contrast's own real image recover its content, within
the region. Reverse R2(eval | train) swaps source/target. Asymmetry = fwd - rev, paired patient
bootstrap CI (same patient set both directions). Healthy-brain reference: same (source,target)
pair, region="healthy".

Outputs: data/synthesis_vs_fill_swap_pooled.csv, tables/synthesis_vs_fill_swap_pooled.md,
plots/synthesis_vs_fill_swap_pooled.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
OUT = THIS_DIR.parent / "outputs"
DATA, TABLES, PLOTS = OUT / "data", OUT / "tables", OUT / "plots"
REGIONS = ["SNFH", "RC", "ET", "NCR"]  # SNFH first, per report instructions
REGION_NAME = {"SNFH": "edema (SNFH)", "RC": "resection cavity (RC)",
               "ET": "enhancing tumor (ET)", "NCR": "necrotic core (NCR)"}
HELP, HURT, NS = "#2f7d6b", "#c0392b", "#9a9a9a"
VARIANT = "raw"  # primary; highpass reported alongside in the table only
N_BOOT, SEED = 5000, 0


def load_r2() -> pd.DataFrame:
    path = DATA / "synthesis_r2_per_patient.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run infer_synth.py first")
    return pd.read_csv(path)


def wide(r2: pd.DataFrame, region: str, variant: str, source: str, target: str) -> pd.Series:
    s = r2[(r2["region"] == region) & (r2["variant"] == variant) &
           (r2["source"] == source) & (r2["target"] == target)]
    return s.set_index("patient")["r2"].dropna()


def boot_mean_ci(x, rng):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan, np.nan
    m = rng.choice(x, size=(N_BOOT, len(x)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def build(r2: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for _, s in sig[sig["family"] == "OOD"].iterrows():
        tr, ev, region = s["train"], s["eval"], s["region"]
        # R2(train|eval): model consumes eval's own real image, predicts train contrast.
        fwd = wide(r2, region, VARIANT, ev, tr)
        rev = wide(r2, region, VARIANT, tr, ev)
        idx = fwd.index.intersection(rev.index)
        fwd, rev = fwd.loc[idx], rev.loc[idx]
        if len(fwd) < 2:
            continue
        fwd_hp = wide(r2, region, "highpass", ev, tr).loc[lambda s: s.index.isin(idx)]
        rev_hp = wide(r2, region, "highpass", tr, ev).loc[lambda s: s.index.isin(idx)]
        h_fwd = wide(r2, "healthy", VARIANT, ev, tr)
        h_rev = wide(r2, "healthy", VARIANT, tr, ev)
        asym = fwd - rev
        f_lo, f_hi = boot_mean_ci(fwd, rng)
        a_lo, a_hi = boot_mean_ci(asym, rng)
        rows.append(dict(
            region=region, train=tr, eval=ev,
            delta_dice_pts=s["mean_delta_pts"], p_holm=s["p_holm"],
            outcome={"HELPS": "success", "HURTS": "failure"}.get(s["direction"], "n.s."),
            n=len(fwd),
            r2_train_given_eval=fwd.mean(), ci_lo=f_lo, ci_hi=f_hi,
            r2_eval_given_train=rev.mean(),
            r2_train_given_eval_hp=fwd_hp.mean() if len(fwd_hp) else np.nan,
            r2_eval_given_train_hp=rev_hp.mean() if len(rev_hp) else np.nan,
            asym=asym.mean(), asym_ci_lo=a_lo, asym_ci_hi=a_hi,
            healthy_train_given_eval=h_fwd.mean() if len(h_fwd) else np.nan,
            healthy_eval_given_train=h_rev.mean() if len(h_rev) else np.nan,
        ))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["region"] = pd.Categorical(out["region"], REGIONS, ordered=True)
    return out.sort_values(["region", "r2_train_given_eval"], ascending=[True, False]).reset_index(drop=True)


def write_md(t: pd.DataFrame, path: Path):
    lines = ["# Real-fill ablation vs. feature-level (learned synthesis) containment R2 (BraTS2024-glioma)", "",
             "**R2(train | eval)** = squared Pearson correlation between a CNN's prediction of the TRAIN "
             "contrast's real content (fed only the EVAL contrast's own real image) and the actual TRAIN-contrast "
             "volume, within the region, z-scored, mean over the 70 eval patients [95% patient-bootstrap CI]. "
             "Hypothesis (Paul, 2026-09-24): high R2(train|eval) -> what the model learned from the training "
             "contrast is recoverable at test time -> real-fill succeeds. **asym** = R2(train|eval) - "
             "R2(eval|train) [paired 95% CI]. `hp` = same quantity on the high-pass (fine-texture) variant. "
             "Ablation: rung4 noise-fill -> rung5 real-fill, Delta Dice points, Holm-corrected over 36 OOD cells.", ""]
    for region in REGIONS:
        sub = t[t["region"] == region]
        if sub.empty:
            continue
        lines += [f"## {REGION_NAME[region]}", "",
                  "| train->eval | Delta Dice | p (Holm) | outcome | R2(train\\|eval) [CI] | R2(eval\\|train) | "
                  "asym [CI] | R2 hp (train\\|eval / eval\\|train) | healthy R2(train\\|eval) |",
                  "|---|--:|--:|---|--:|--:|--:|--:|--:|"]
        for _, r in sub.iterrows():
            oc = {"success": "**success**", "failure": "**FAILURE**"}.get(r["outcome"], "n.s.")
            lines.append(
                f"| {r['train']}->{r['eval']} | {r['delta_dice_pts']:+.1f} | {r['p_holm']:.2g} | {oc} | "
                f"{r['r2_train_given_eval']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | "
                f"{r['r2_eval_given_train']:.3f} | {r['asym']:+.3f} [{r['asym_ci_lo']:+.3f}, {r['asym_ci_hi']:+.3f}] | "
                f"{r['r2_train_given_eval_hp']:.3f} / {r['r2_eval_given_train_hp']:.3f} | "
                f"{r['healthy_train_given_eval']:.3f} |")
        lines.append("")
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def plot(t: pd.DataFrame, path: Path):
    regions_present = [r for r in REGIONS if (t["region"] == r).any()]
    fig, axes = plt.subplots(1, len(regions_present), figsize=(4.4 * len(regions_present), 5.4), sharex=True)
    if len(regions_present) == 1:
        axes = [axes]
    col = {"success": HELP, "failure": HURT, "n.s.": NS}
    mk = {"success": "^", "failure": "v", "n.s.": "o"}
    for ax, region in zip(axes, regions_present):
        sub = t[t["region"] == region].iloc[::-1]
        for yi, (_, r) in enumerate(sub.iterrows()):
            c = col[r["outcome"]]
            ax.plot([r["r2_eval_given_train"], r["r2_train_given_eval"]], [yi, yi],
                    color=c, alpha=0.35, linewidth=3, solid_capstyle="round")
            ax.plot([r["ci_lo"], r["ci_hi"]], [yi, yi], color=c, linewidth=1.2)
            ax.scatter(r["r2_eval_given_train"], yi, s=36, facecolors="white", edgecolors=c,
                       linewidths=1.4, zorder=3)
            ax.scatter(r["r2_train_given_eval"], yi, s=80 if r["outcome"] != "n.s." else 45,
                       marker=mk[r["outcome"]], color=c, edgecolors="white", linewidths=1, zorder=4)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels([f"{a}->{b}  ({d:+.1f})" for a, b, d in
                            zip(sub["train"], sub["eval"], sub["delta_dice_pts"])], fontsize=8)
        ax.set_title(REGION_NAME[region], fontsize=10)
        ax.set_xlabel("R2 (pooled, raw)", fontsize=8.5)
        ax.grid(axis="x", color="#eee")
        ax.tick_params(axis="x", labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles = [plt.Line2D([], [], marker="^", color=HELP, ls="", ms=8, label="R2(train|eval), real-fill success"),
               plt.Line2D([], [], marker="v", color=HURT, ls="", ms=8, label="R2(train|eval), real-fill failure"),
               plt.Line2D([], [], marker="o", color=NS, ls="", ms=6, label="R2(train|eval), n.s."),
               plt.Line2D([], [], marker="o", color="#555", mfc="white", ls="", ms=6, label="reverse R2(eval|train)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("Is the training contrast's content recoverable from the eval contrast (learned synthesis)?\n"
                 "Filled = R2(train|eval) with 95% CI, hollow = reverse; Delta Dice pts in brackets",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.10, 1, 0.90))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    for d in (DATA, TABLES, PLOTS):
        d.mkdir(parents=True, exist_ok=True)
    r2 = load_r2()
    print(f"Loaded {len(r2)} rows, {r2['patient'].nunique()} patients")
    sig = pd.read_csv(DATA / "region_fill_swap_significance.csv")
    t = build(r2, sig)
    if t.empty:
        raise SystemExit("no rows built -- check synthesis_r2_per_patient.csv has data for the OOD cells")
    t.to_csv(DATA / "synthesis_vs_fill_swap_pooled.csv", index=False)
    write_md(t, TABLES / "synthesis_vs_fill_swap_pooled.md")
    plot(t, PLOTS / "synthesis_vs_fill_swap_pooled.png")
    pd.set_option("display.width", 220)
    print(t[["region", "train", "eval", "delta_dice_pts", "outcome", "n", "r2_train_given_eval",
             "r2_eval_given_train", "asym", "asym_ci_lo", "asym_ci_hi",
             "healthy_train_given_eval"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Explicit checks of the hypothesis's stated predictions, printed for the report.
    def cell(train, ev, region):
        r = t[(t["train"] == train) & (t["eval"] == ev) & (t["region"] == region)]
        return None if r.empty else r.iloc[0]

    print("\n=== Hypothesis predictions (SNFH, unless noted) ===")
    for train, ev in [("t1n", "t2w"), ("t1n", "t2f")]:
        f = cell(train, ev, "SNFH")
        r = cell(ev, train, "SNFH")
        if f is not None and r is not None:
            print(f"R2({train}|{ev})={f['r2_train_given_eval']:.3f} vs R2({ev}|{train})={r['r2_train_given_eval']:.3f}"
                  f"  [predicted: former > latter]  -> {'HOLDS' if f['r2_train_given_eval'] > r['r2_train_given_eval'] else 'DOES NOT HOLD'}")
    c1 = cell("t2w", "t1c", "SNFH")
    if c1 is not None:
        print(f"R2(t2w|t1c)={c1['r2_train_given_eval']:.3f} (should be relatively high; compare to failures below)")
    for train, ev, region in [("t2w", "t1n", "SNFH"), ("t2f", "t1n", "SNFH"),
                               ("t2w", "t1n", "RC"), ("t2f", "t1c", "RC")]:
        c = cell(train, ev, region)
        if c is not None:
            print(f"[failure] {train}->{ev} {region}: R2(train|eval)={c['r2_train_given_eval']:.3f} "
                  f"delta_dice={c['delta_dice_pts']:+.1f}")


if __name__ == "__main__":
    main()
