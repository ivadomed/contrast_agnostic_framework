#!/usr/bin/env python3
"""
fig:ngf-dose -- the noise-vs-real NGF analysis: downstream OOD Dice (left) and
texture fidelity measured WITHOUT any segmentation (right) across the same
ablation rungs, on Open-MS FLAIR.

Why this figure exists: the fill-swap rung changes Dice, but a reader can fairly
ask whether it changed *texture* or something else. NGF answers that directly.
The noise-fill rungs should sit at the texture floor while Dice still climbs
(their gains come from contrast randomization, not texture); only the real-fill
swap should lift both together.

Sources (both read live, nothing transcribed):
  * Dice  -- Open-MS FLAIR ladder_series.json, written by the shared engine
             (00_commun_scripts/00_03_evaluate/ladder_ood_common.py). Read from
             disk so the panel tracks every engine fix (e.g. the 2026-10-01
             fold-3 cap, which changed this ladder).
  * NGF   -- open-ms texture_analysis_lvl_1 `ngf_flair_noblur_eroded` set:
             foreground ROI, eroded to drop the region-boundary shell where a
             piecewise-constant fill still has gradients. "palette" in that data
             is v26_6_2 alone (see that dir's FINDINGS.md), i.e. exactly the
             real-fill rung, and v26_6_2_noisefill_v2 is the noise-fill rung.

Rebuilt 2026-10-02: the previous version of this figure was produced ad hoc on
2026-08-02 with no script in the repo, so it could not be regenerated after the
ladder changed underneath it.

Usage:
  .venv/bin/python make_ngf_dose_response.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
OPENMS = REPO / "benchmark/02_tasks/brain_ms/open-ms"
LADDER = OPENMS / "8_results_open-ms/02_metrics/open_ms_model/flair/ablations/ladder_series.json"
NGF_GLOB = str(OPENMS / "7_analysis_open-ms/texture_analysis_lvl_1/outputs/data"
               / "ngf_flair_noblur_eroded_rank*.csv")
OUT = REPO / "paper/cvpr_format_latex/figures/ngf_dose_response_openms_flair"

# (axis label, ladder rung label, NGF method id). Rung labels are matched by the
# engine's own strings so a reordered ladder cannot silently misalign the panels.
RUNGS = [
    ("+$k$-means",    "+kmeans",                "baseline_kmeans"),
    ("+label remap",  "+label_remap",           "baseline_kmeans_label_remap"),
    ("+Voronoi\n(noise fill)", "+voronoi (noise fill)", "v26_6_2_noisefill_v2"),
    ("real fill\n(PALETTE)",   "v26_6_2 (real fill)",   "palette"),
]
NOISE_FLOOR = 1.0 / 3.0          # NGF of isotropic noise
NOISE, REAL = "#8a8a8a", "#2f7d6b"   # the paper's FLAT / IMPROVE colours


def main():
    d = json.loads(LADDER.read_text())
    dice = dict(zip(d["labels"], d["dice"]))
    missing = [r for _, r, _ in RUNGS if r not in dice]
    if missing:
        raise SystemExit(f"ladder lacks rung(s) {missing}; labels are {d['labels']}")

    ngf = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(NGF_GLOB)))
    ngf = ngf[ngf.roi_id == "foreground"]
    # One value per subject first (averaging that subject's generated variants),
    # so the error bar is across the 30 independent scans, not across variants.
    per_subj = ngf.groupby(["method", "subject"])["ngf_all"].mean()

    x = np.arange(len(RUNGS))
    colors = [NOISE] * (len(RUNGS) - 1) + [REAL]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.7))

    a1.plot(x, [dice[r] for _, r, _ in RUNGS], color="#333333", lw=1.4, zorder=1)
    a1.scatter(x, [dice[r] for _, r, _ in RUNGS], c=colors, s=46, zorder=2,
               edgecolor="white", linewidth=0.8)
    a1.set_ylabel("OOD Dice (%)")
    a1.set_title("Downstream segmentation", fontsize=10)

    m = [per_subj[mid].mean() for _, _, mid in RUNGS]
    se = [per_subj[mid].std(ddof=1) / np.sqrt(per_subj[mid].size) for _, _, mid in RUNGS]
    a2.errorbar(x, m, yerr=se, color="#333333", lw=1.4, capsize=2, zorder=1)
    a2.scatter(x, m, c=colors, s=46, zorder=2, edgecolor="white", linewidth=0.8)
    a2.axhline(NOISE_FLOOR, color=NOISE, ls=":", lw=1)
    a2.text(0.02, NOISE_FLOOR + 0.012, "isotropic-noise floor", color=NOISE,
            fontsize=7.5, transform=a2.get_yaxis_transform())
    a2.set_ylabel("NGF similarity to source")
    a2.set_title("Texture fidelity (no segmentation)", fontsize=10)

    for ax in (a1, a2):
        ax.set_xticks(x)
        ax.set_xticklabels([l for l, _, _ in RUNGS], fontsize=7.5)
        ax.axvspan(len(RUNGS) - 1.5, len(RUNGS) - 0.5, color="#f0c96b", alpha=0.35, lw=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=300)
        print("wrote", f"{OUT}.{ext}")
    print("Dice  :", {r: round(dice[r], 2) for _, r, _ in RUNGS})
    print("NGF   :", {mid: round(v, 3) for (_, _, mid), v in zip(RUNGS, m)},
          f"(n={per_subj['palette'].size} subjects)")


if __name__ == "__main__":
    main()
