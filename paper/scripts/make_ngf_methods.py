#!/usr/bin/env python3
"""
NGF texture fidelity of every augmentation method on the 7 ablation tasks (paper fig:ngf-methods
+ supplementary tab:ngf). Input: paper/generated_results/ngf_all_methods/ngf_per_draw.csv.gz
(scripts/ngf_all_methods.py; variant `noblur` = blur/resolution degradation off so NGF reflects
the fill, ROI `fg` = foreground eroded x3). Unit = one source scan (its draws averaged).
Task value = mean over the task's contrasts of the per-contrast mean over scans.
Test: PALETTE alone vs each other method, per task, paired Wilcoxon over scans (one-sided,
PALETTE alone higher), Holm over all (task, method) pairs.

Writes  cvpr_format_latex/figures/ngf_methods.{pdf,png}
        cvpr_format_latex/sec/_suppl_ngf_table.tex   (generated -- do not hand-edit)
        generated_results/ngf_all_methods/ngf_methods_tests.csv
Usage:  .venv/bin/python paper/scripts/make_ngf_methods.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

PAPER = Path(__file__).resolve().parent.parent
SRC = PAPER / "generated_results/ngf_all_methods/ngf_per_draw.csv.gz"
FIG = PAPER / "cvpr_format_latex/figures/ngf_methods"
TEX = PAPER / "cvpr_format_latex/sec/_suppl_ngf_table.tex"
TESTS = PAPER / "generated_results/ngf_all_methods/ngf_methods_tests.csv"

TASKS = [("glioma", "Glioma", True), ("ms", "MS", True), ("breast", "Breast", True),
         ("abdomen", "Abdomen", False), ("brain_healthy", "Brain", False),
         ("mandible", "Mandible", False), ("pelvis", "Pelvis", False)]   # True = appearance-defined target
METHODS = [("palette_train050", "PALETTE alone"), ("ours_train050", "PALETTE-Aug"), ("auglab_default", "Auglab"),
           ("srcsm", "SRCSM"), ("synthseg_EM", "SynthSeg-EM"), ("synthseg_noEM", "SynthSeg-noEM"),
           ("noisefill_lblvor_train050", "noise fill")]
# REAL USE (Paul, 2026-10-08): every row is measured with the method's TRAINING configuration, synthesis probability
# included -- PALETTE alone and the noise fill (rung 4, label_voronoi) at their training probability 0.5 (keys *_train050),
# like PALETTE-Aug (0.5) and the competitors (their own recipes). The forced-synthesis keys (palette, noisefill_lblvor:
# probability 1) stay in the per-draw data for reference only. REFS = the two reference rows written per scan.
REFS = [("identity", "source image (no augmentation)"), ("noise", "isotropic noise")]
REF = "palette_train050"
REAL, REAL_LIGHT, NOISE, OTHER = "#2f7d6b", "#8fc4b4", "#8a8a8a", "#5b7083"
NOISE_FLOOR = 1 / 3
plt.rcParams.update({"font.size": 7.5, "font.family": "sans-serif", "axes.edgecolor": "#555555",
                     "axes.linewidth": 0.8, "xtick.color": "#333333", "ytick.color": "#333333"})


def load() -> pd.DataFrame:
    d = pd.read_csv(SRC)
    d = d[(d.variant == "noblur") & (d.roi == "fg") & d.method.isin([m for m, _ in METHODS])]
    s = d.groupby(["task", "contrast", "scan", "method"], as_index=False).ngf_all.mean()   # draws -> scan
    missing = {(t, m) for t, _, _ in TASKS for m, _ in METHODS} - set(zip(s.task, s.method))
    if missing:
        sys.exit(f"missing (task, method) in {SRC}: {sorted(missing)}")
    return s


def ref_task_means() -> pd.DataFrame:
    """Task means of the per-scan reference rows (variant 'ref'): identity = NGF of the source to itself (1 by
    construction), noise = i.i.d. uniform noise (the isotropic floor, ~1/3)."""
    d = pd.read_csv(SRC)
    d = d[(d.variant == "ref") & (d.roi == "fg") & d.method.isin([m for m, _ in REFS])]
    s = d.groupby(["task", "contrast", "scan", "method"], as_index=False).ngf_all.mean()
    return task_means(s)


def task_means(s: pd.DataFrame) -> pd.DataFrame:
    c = s.groupby(["task", "contrast", "method"]).ngf_all.mean().reset_index()
    return c.groupby(["task", "method"]).ngf_all.mean().unstack("method")


def tests(s: pd.DataFrame) -> pd.DataFrame:
    w = s.pivot_table(index=["task", "contrast", "scan"], columns="method", values="ngf_all")
    rows = []
    for t, _, _ in TASKS:
        x = w.loc[t]
        for m, _ in METHODS:
            if m == REF:
                continue
            p = x[[REF, m]].dropna()
            rows.append({"task": t, "method": m, "n_scans": len(p), "delta": float((p[REF] - p[m]).mean()),
                         "p": float(wilcoxon(p[REF], p[m], alternative="greater").pvalue)})
    r = pd.DataFrame(rows)
    order = np.argsort(r.p.values)            # Holm step-down
    adj, run = np.empty(len(r)), 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (len(r) - i) * r.p.values[j]))
        adj[j] = run
    r["p_holm"] = adj
    return r


def figure(s: pd.DataFrame, tm: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    labels = []
    for i, (m, name) in enumerate(METHODS):
        y = len(METHODS) - 1 - i
        col = REAL if m == REF else REAL_LIGHT if m == "ours_train050" else NOISE if m == "noisefill_lblvor_train050" else OTHER
        v = ax.violinplot(s[s.method == m].ngf_all.values, positions=[y], vert=False, widths=0.82,
                          showextrema=False)
        for b in v["bodies"]:
            b.set_facecolor(col); b.set_edgecolor("none"); b.set_alpha(0.35)
        for t, _, appearance in TASKS:
            ax.scatter(tm.loc[t, m], y, s=13, zorder=3, linewidth=0.9,
                       facecolor=col if appearance else "white", edgecolor=col)
        mean = float(tm[m].mean())
        ax.plot([mean, mean], [y - 0.32, y + 0.32], color=col, lw=1.6, zorder=4)
        ax.text(1.005, y, f"{mean:.2f}", va="center", ha="left", fontsize=7,
                color=REAL if m == "ours_train050" else col,
                transform=ax.get_yaxis_transform())
        labels.append(name)
    ax.axvline(NOISE_FLOOR, color=NOISE, ls=":", lw=0.9)
    ax.axvline(1.0, color=REAL, ls=":", lw=0.9)
    ax.text(1.0 - 0.004, len(METHODS) - 2.5, "source image (no aug.)", color=REAL, fontsize=6,
            rotation=90, ha="right", va="center")
    ax.text(NOISE_FLOOR - 0.004, len(METHODS) - 2.5, "isotropic-noise floor", color=NOISE, fontsize=6,
            rotation=90, ha="right", va="center")
    ax.set_yticks(range(len(METHODS))); ax.set_yticklabels(labels[::-1])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xlim(0.28, 1.005); ax.set_ylim(-0.6, len(METHODS) - 0.2)
    ax.set_xlabel("NGF to the source scan (1 = texture fully kept)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    h = [plt.Line2D([], [], ls="", marker="o", ms=3.6, mfc="#444444", mec="#444444"),
         plt.Line2D([], [], ls="", marker="o", ms=3.6, mfc="white", mec="#444444"),
         plt.Line2D([], [], color="#444444", lw=1.6)]
    fig.legend(h, ["appearance-defined task", "interface-bounded task", "mean"],
              loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6.2, frameon=False,
              handletextpad=0.25, columnspacing=0.8, borderaxespad=0.1)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.94))
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG}.{ext}", dpi=300)


def tex_p(p: float) -> str:
    m, e = f"{p:.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def table(tm: pd.DataFrame, r: pd.DataFrame) -> None:
    best = {t: max(tm.loc[t, m] for m, _ in METHODS) for t, _, _ in TASKS}
    sig = r[r.p_holm < 0.05]
    worst_p = sig.p_holm.max()
    tname, mname = {t: n for t, n, _ in TASKS}, dict(METHODS)
    ns = [f"{mname[x.method]} on {tname[x.task]} ($\\Delta={x.delta:+.3f}$, $p={x.p_holm:.2f}$)"
          for x in r[r.p_holm >= 0.05].itertuples()]
    claim = ("PALETTE alone keeps more texture than every other method on every task"
             if not ns else "PALETTE alone keeps more texture than every other method on every task, except "
             + "; ".join(ns) + ", where the difference is not significant")
    lines = ["% GENERATED by paper/scripts/make_ngf_methods.py -- do not hand-edit",
             "\\begin{table}[h]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3.5pt}",
             "\\resizebox{\\linewidth}{!}{%", "\\begin{tabular}{l" + "c" * (len(TASKS) + 1) + "}", "\\toprule",
             "Method & " + " & ".join(n for _, n, _ in TASKS) + " & Mean \\\\", "\\midrule"]
    for m, name in METHODS:
        cells = []
        for t, _, _ in TASKS:
            v = tm.loc[t, m]
            cells.append(f"\\textbf{{{v:.3f}}}" if np.isclose(v, best[t]) else f"{v:.3f}")
        mean = tm[m].mean()
        cells.append(f"\\textbf{{{mean:.3f}}}" if m == REF else f"{mean:.3f}")
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    rt = ref_task_means()
    lines.append("\\midrule")
    for m, name in REFS:
        lines.append(f"\\emph{{{name}}} & " + " & ".join(f"{rt.loc[t, m]:.3f}" for t, _, _ in TASKS)
                     + f" & {rt[m].mean():.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}",
              "\\caption{Texture fidelity per task: NGF similarity of the augmented volume to its source scan "
              "(1 = all texture kept, $\\approx\\!1/3$ = isotropic noise) on the seven tasks with a ladder "
              "(Spine has none), mean over contrasts of the mean over "
              f"scans ($16$--$20$ per contrast, $5$ draws each). Every method is measured as used in training, synthesis "
              "probability included: PALETTE alone (the ladder's real-fill rung), the noise fill and PALETTE-Aug "
              "synthesize in half of the draws, SRCSM and SynthSeg in every draw; a draw without synthesis keeps the "
              "source intensities (PALETTE alone, noise fill) or gets Auglab's own operators (PALETTE-Aug). Reference "
              "rows: the source image itself (no augmentation) and i.i.d. noise. "
              "PALETTE-Aug applies PALETTE in half of its draws, first in the Auglab chain, whose operators then "
              "act on the remapped image: the combination keeps less texture than either PALETTE alone or Auglab. "
              f"{claim} (paired one-sided Wilcoxon over scans, "
              f"Holm-corrected over the ${len(r)}$ comparisons; largest significant $p={tex_p(worst_p)}$).}}",
              "\\label{tab:ngf}", "\\end{table}"]
    TEX.write_text("\n".join(lines) + "\n")


def main():
    s = load()
    tm = task_means(s)
    r = tests(s)
    r.to_csv(TESTS, index=False)
    figure(s, tm)
    table(tm, r)
    print(tm[[m for m, _ in METHODS]].round(3).rename(columns=dict(METHODS)).to_string())
    print("\nmean over tasks:", {n: round(float(tm[m].mean()), 3) for m, n in METHODS})
    print("\nPALETTE alone > X  (Holm p, max per method):", r.groupby("method").p_holm.max().to_dict())
    print("not significant (Holm >= .05):", r[r.p_holm >= 0.05][["task", "method", "delta", "p_holm"]].to_dict("records"))


if __name__ == "__main__":
    main()
