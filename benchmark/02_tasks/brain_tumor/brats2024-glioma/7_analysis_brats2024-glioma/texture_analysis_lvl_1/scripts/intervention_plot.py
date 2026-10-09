#!/usr/bin/env python
"""Per-patient before/after (orig -> intervened) edema Dice plot, noise-fill vs real-fill,
one panel per pre-registered experiment. Reads outputs/data/intervention_per_patient.csv
(written by intervention_analyze.py). CPU-only, tiny (30 patients x 4 experiments)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parents[1]  # .../texture_analysis_lvl_1
DATA = ANALYSIS_DIR / "outputs" / "data" / "intervention_per_patient.csv"
OUT = ANALYSIS_DIR / "outputs" / "plots" / "intervention_before_after_dice.png"

EXPERIMENTS = [
    ("E1_flatten_t2f_t1n_trained", "E1: t1n-trained,\nFLATTEN(t2f)"),
    ("E2_flatten_t2f_t2w_trained_cross", "E2 cross: t2w-trained,\nFLATTEN(t2f)"),
    ("E2_flatten_t2w_t2w_trained_indomain", "E2 in-domain: t2w-trained,\nFLATTEN(t2w)"),
    ("E3_ramp_t1n_donor_t2w_t2w_trained", "E3: t2w-trained,\nRAMP(t1n, donor=t2w)"),
]


def main():
    df = pd.read_csv(DATA)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    for ax, (exp, title) in zip(axes, EXPERIMENTS):
        sub = df[df["experiment"] == exp]
        for rung, color, dx in (("noise", "#d95f02", -0.08), ("real", "#1b9e77", 0.08)):
            r = sub[sub["rung"] == rung].sort_values("case")
            for _, row in r.iterrows():
                x0, x1 = 0 + dx, 1 + dx
                ax.plot([x0, x1], [row["dice_orig"], row["dice_interv"]],
                         color=color, alpha=0.4, linewidth=1)
            ax.scatter([0 + dx] * len(r), r["dice_orig"], color=color, s=12, alpha=0.7)
            ax.scatter([1 + dx] * len(r), r["dice_interv"], color=color, s=12, alpha=0.7,
                       label=f"{rung}-fill" if ax is axes[0] else None)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["orig", "intervened"])
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("edema (SNFH) Dice")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Per-patient edema Dice, before -> after intervention (fold 0, n=30 per experiment)")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
