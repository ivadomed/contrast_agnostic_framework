#!/usr/bin/env python
"""
Per-LABEL view of the two totalseg-pelvic causal-ablation ladders (ct-trained, mri-trained): which structure each rung
helps. Reads the rung run dirs from each ladder's own ladder_series.json (run_keys), folds 0-2 eval_all.csv, mean Dice
per (test contrast, label, rung). Bespoke analysis (non-canonical 06_2X slot), 2026-10-07.
Outputs (8_results_totalseg-pelvic/02_metrics/totalseg_pelvic_model/ladder_per_label/):
  ladder_per_label_curves.png   2x2 small multiples (arm x test contrast), one curve per label
  ladder_per_label_deltas.png   per label: rung 4->5 (real fill - noise fill) and 5->6 (+AugLab) Dice deltas, OOD directions
  ladder_per_label.md / .csv    the numbers
Usage: .venv/bin/python 06_20_ladder_per_label_viz.py   (CPU, light; submit through run_job)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2] / "8_results_totalseg-pelvic/02_metrics/totalseg_pelvic_model"
OUT = ROOT / "ladder_per_label"; OUT.mkdir(exist_ok=True)
ARMS = ("ct", "mri")
STRUCT = ["gluteus_maximus", "gluteus_medius", "gluteus_minimus", "hip", "iliopsoas", "sacrum"]
COLOR = dict(zip(STRUCT, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]))  # validated palette
SHORT = {"+voronoi (noise fill)": "+voronoi\n(noise)", "v26_6_2 (real fill)": "real fill", "+AugLab (OURS)": "+AugLab", "baseline (floor)": "baseline", "+kmeans": "+kmeans", "+label_remap": "+remap"}
POS, NEG, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"


def load(arm):
    d = json.loads((ROOT / arm / "ablations/ladder_series.json").read_text())
    tabs = {}
    for lab, key in zip(d["labels"], d["run_keys"]):
        name = Path(key).name
        cands = [p for p in (ROOT / arm).rglob(f"*{name}") if p.is_dir() and (p / "fold0/eval_all.csv").exists()]
        assert cands, f"{arm}: no metrics dir for {key}"
        df = pd.concat([pd.read_csv(f) for f in sorted(cands[0].glob("fold[0-2]/eval_all.csv"))])
        tabs[lab] = df.groupby(["group", "label"])["dice"].mean() * 100
    return d["labels"], pd.DataFrame(tabs)


def struct_side(label):
    for s in STRUCT:
        if label.startswith(s):
            return s, ("R" if label.endswith("_right") else "L" if label.endswith("_left") else "")
    return label, ""


def main():
    rungs, T = {}, {}
    for arm in ARMS:
        rungs[arm], T[arm] = load(arm)
    rows = []
    for arm in ARMS:
        for grp in sorted(set(T[arm].index.get_level_values(0))):
            t = T[arm].loc[grp]
            for lab, r in t.iterrows():
                rows.append(dict(train=arm, test=grp, ood=grp != arm, label=lab, **{k: float(v) for k, v in r.items()},
                                 d45=r["v26_6_2 (real fill)"] - r["+voronoi (noise fill)"], d56=r["+AugLab (OURS)"] - r["v26_6_2 (real fill)"],
                                 d16=r["+AugLab (OURS)"] - r["baseline (floor)"]))
    L = pd.DataFrame(rows); L.to_csv(OUT / "ladder_per_label.csv", index=False)
    with open(OUT / "ladder_per_label.md", "w") as f:
        f.write("# totalseg-pelvic ladders per label (mean Dice, folds 0-2)\n\n")
        for arm in ARMS:
            for grp in sorted(set(T[arm].index.get_level_values(0))):
                t = T[arm].loc[grp].copy(); t["4->5"] = t["v26_6_2 (real fill)"] - t["+voronoi (noise fill)"]; t["5->6"] = t["+AugLab (OURS)"] - t["v26_6_2 (real fill)"]
                f.write(f"## trained {arm}, test {grp}{' (OOD)' if grp != arm else ' (in-domain)'}\n\n" + "```\n" + t.round(1).to_string() + "\n```\n\n")

    # ---- figure 1: curves
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharey=True, facecolor=SURF)
    for i, arm in enumerate(ARMS):
        x = np.arange(len(rungs[arm]))
        for j, grp in enumerate([g for g in ARMS if g != arm] + [arm]):
            ax = axes[i, j]; ax.set_facecolor(SURF)
            t = T[arm].loc[grp]
            ends = {}
            for lab, r in t.iterrows():
                s, side = struct_side(lab)
                ax.plot(x, r.values, color=COLOR[s], lw=2, ls="-" if side != "R" else (0, (4, 2)), marker="o", ms=4, mfc=SURF, mew=1.5, mec=COLOR[s])
                if side != "R":
                    ends[s] = float(r.values[-1])
            # direct labels, pushed apart so they never overlap (min 3.5 Dice points between neighbours)
            order_s = sorted(ends, key=ends.get); ys = [ends[s] for s in order_s]
            for k in range(1, len(ys)):
                ys[k] = max(ys[k], ys[k - 1] + 3.5)
            for s, yy in zip(order_s, ys):
                ax.annotate(s.replace("_", " "), (x[-1] + 0.08, yy), color=COLOR[s], fontsize=8, va="center", ha="left")
            ax.set_xticks(x); ax.set_xticklabels([SHORT.get(r, r) for r in rungs[arm]], fontsize=8)
            ax.set_title(f"trained on {arm.upper()}  →  tested on {grp.upper()}  ({'OOD' if grp != arm else 'in-domain'})", fontsize=10, color=INK, loc="left")
            ax.grid(axis="y", color="#e6e5e2", lw=0.8); ax.set_axisbelow(True)
            for sp in ("top", "right"): ax.spines[sp].set_visible(False)
            ax.set_xlim(-0.3, len(x) - 0.2 + 1.3)
            ax.axvspan(2.5, 4.5, color="#f0efec", zorder=0)
            if j == 0: ax.set_ylabel("Dice (%)")
    axes[0, 0].annotate("shaded = rung 4→5, the fill swap (noise → real texture)", (0.01, 0.98), xycoords="axes fraction", fontsize=8, color=MUTED, va="top")
    h = [plt.Line2D([], [], color=MUTED, lw=2, ls="-", label="left"), plt.Line2D([], [], color=MUTED, lw=2, ls=(0, (4, 2)), label="right")]
    h += [plt.Line2D([], [], color=COLOR[s], lw=2, label=s.replace("_", " ")) for s in STRUCT]
    fig.legend(handles=h, loc="lower center", ncol=8, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("totalseg-pelvic causal-ablation ladders, per label (mean Dice over folds 0-2; rung 5 = val000 retrain)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96)); fig.savefig(OUT / "ladder_per_label_curves.png", dpi=150, facecolor=SURF); plt.close(fig)

    # ---- figure 2: deltas, OOD directions
    O = L[L.ood].copy(); O["struct"] = [struct_side(l)[0] for l in O.label]; O["side"] = [struct_side(l)[1] for l in O.label]
    order = [l for s in STRUCT for l in O.label.unique() if l.startswith(s)]
    fig, axes = plt.subplots(1, 4, figsize=(13, 5.2), sharey=True, facecolor=SURF)
    panels = [("ct", "d45", "CT-trained → MRI\nrung 4→5: real fill − noise fill"), ("ct", "d56", "CT-trained → MRI\nrung 5→6: +AugLab"),
              ("mri", "d45", "MRI-trained → CT\nrung 4→5: real fill − noise fill"), ("mri", "d56", "MRI-trained → CT\nrung 5→6: +AugLab")]
    y = np.arange(len(order))[::-1]
    for ax, (arm, col, title) in zip(axes, panels):
        ax.set_facecolor(SURF)
        v = O[O.train == arm].set_index("label").loc[order, col].values
        ax.barh(y, v, color=[POS if a >= 0 else NEG for a in v], height=0.62)
        for yi, a in zip(y, v):
            ax.annotate(f"{a:+.1f}", (a, yi), xytext=(4 if a >= 0 else -4, 0), textcoords="offset points", ha="left" if a >= 0 else "right", va="center", fontsize=8, color=INK)
        ax.axvline(0, color=MUTED, lw=0.8); ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.grid(axis="x", color="#e6e5e2", lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
        lim = max(abs(v).max() * 1.35, 4); ax.set_xlim(-lim, lim); ax.set_xlabel("Δ Dice (points)")
    axes[0].set_yticks(y); axes[0].set_yticklabels([l.replace("_", " ") for l in order], fontsize=8.5)
    for yt, l in zip(axes[0].get_yticklabels(), order): yt.set_color(COLOR[struct_side(l)[0]])
    fig.suptitle("Which labels the texture step and the AugLab step help — OOD directions only (mean Dice over folds 0-2)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(OUT / "ladder_per_label_deltas.png", dpi=150, facecolor=SURF); plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
